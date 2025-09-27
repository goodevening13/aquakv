import math
from argparse import Namespace
from typing import Sequence, Optional, List, Tuple

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm, trange

from aquakv import datautils, modelutils
from aquakv.quantizers import QuantizerBase, HiggsQuantizer
from aquakv.linear_utils import fit_linear_regression
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

class OutputCatcher(nn.Module):
    """Wraps a layer to catch its output tensors from one or more forward passes"""

    def __init__(self, inner: nn.Module, offload_activations: bool):
        super().__init__()
        self.inner = inner
        self.offload_activations = offload_activations
        self.outputs = []

    def forward(self, inp, **kwargs):
        output = self.inner(inp)
        self.outputs.append(output.to('cpu' if self.offload_activations else inp.device, copy=True))
        return output


def get_predictor(args: Namespace, predictor_inputs: List[torch.Tensor], targets: List[torch.Tensor]
                  ) -> Tuple[nn.Module, float, float]:
    """Train predictor, return the predictor itself and train/valid errors"""
    device = args.devices[0]
    X = torch.stack(predictor_inputs, dim=0)
    Y = torch.stack(targets, dim=0)
    assert X.shape[:-1] == Y.shape[:-1] == (args.total_nsamples, args.model_seqlen)
    train_ids, valid_ids = torch.randperm(
        len(X), generator=torch.Generator(X.device).manual_seed(args.seed), device="cpu"
    ).split_with_sizes((args.total_nsamples - args.valid_nsamples, args.valid_nsamples))
    X_train, X_valid, Y_train, Y_valid = X[train_ids], X[valid_ids], Y[train_ids], Y[valid_ids]
    X_train, X_valid, Y_train, Y_valid = [t.flatten(0, -2) for t in (X_train, X_valid, Y_train, Y_valid)]
    weight, bias = fit_linear_regression(
        X_train, Y_train, reg_rate=args.percdamp, fit_intercept=True,
        compute_device=device, compute_dtype=torch.float32, chunk_size=args.chunk_size
    )
    predictor = nn.Linear(*weight.shape[::-1], dtype=X.dtype, device=device)
    with torch.no_grad():
        predictor.weight[...] = weight
        predictor.bias[...] = bias
    mse_train = compute_relative_mse(
        predictor, X_train, Y_train, compute_device=device, chunk_size=args.chunk_size)
    mse_valid = compute_relative_mse(
        predictor, X_valid, Y_valid, compute_device=device, chunk_size=args.chunk_size)
    return predictor, mse_train, mse_valid

def compute_causal_mask(q_len, k_len, device):
    mask = torch.tril(torch.ones((1, 1, q_len, k_len), dtype=torch.float32, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


@torch.no_grad()
def attention_weights_error(keys, reference_next_layers_keys, next_layer_queries, model, layer, predictor, args):
    bsz = keys.shape[0]
    device = args.devices[0]
    _, valid_ids = torch.randperm(
        bsz, generator=torch.Generator("cpu").manual_seed(args.seed), device="cpu"
    ).split_with_sizes((args.total_nsamples - args.valid_nsamples, args.valid_nsamples))
    valid_ids = valid_ids.to(device)
    predicted_next_layer_keys = predictor(keys[valid_ids])
    
    reference_attention_score = compute_attention_score(model, layer, reference_next_layers_keys[valid_ids], next_layer_queries[valid_ids], device)
    predicted_attention_score = compute_attention_score(model, layer, predicted_next_layer_keys, next_layer_queries[valid_ids], device)
    print("attention score shapes", reference_attention_score.shape, predicted_attention_score.shape)

    # [16, 24, 8192, 8192]
    rmse_heads = []
    stds = []
    mask = torch.tril(torch.ones((1, predicted_attention_score.shape[1], reference_attention_score.shape[2], reference_attention_score.shape[3]), dtype=torch.int))
    non_zero_elems = mask.sum()
    for i in tqdm(range(args.valid_nsamples), desc="computing rmse and std"):
        xb, yb = [tensor[i: i + 1].to(device=device, non_blocking=True) for tensor in (predicted_attention_score, reference_attention_score)]
        rmse_head = torch.square(xb - yb)[mask.bool()].sum() / non_zero_elems # [b, n_heads, q_len, q_len] = [1, 24, 8k, 8k] => [24]
        std = rmse_head / yb[mask.bool()].std()
        stds.append(std)
        rmse_heads.append(rmse_head)
    rmse_batch_heads = torch.stack(rmse_heads) # [16, 24]
    stds = torch.stack(stds)
    rmse_heads = rmse_batch_heads.sum(dim=0) / args.valid_nsamples # [16, 24] => [24]
    std = stds.sum(dim=0) / args.valid_nsamples
    print(rmse_heads)
    print(std)

@torch.no_grad()
def compute_attention_score(model, layer, keys, queries, device):
    bsz, q_len = queries.shape[0], queries.shape[2]
    position_ids = torch.arange(q_len).unsqueeze(dim=0)
    position_embeddings = model.model.rotary_emb(keys, position_ids)
    cos, sin = position_embeddings
    sin = sin.to(device=device)
    cos = cos.to(device=device)
    query_states = queries.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
    key_states = keys.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)
    attn_weights = []
    causal_mask = compute_causal_mask(q_len, q_len, device=key_states.device)
    for i in tqdm(range(bsz), desc="computing attention scores"):
        attn_weight = torch.matmul(query_states[i: i + 1], key_states[i: i + 1].transpose(2, 3)) / math.sqrt(layer.self_attn.head_dim) # [b, n_heads, q_len, k_len]
        attn_weight += causal_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weight = attn_weight.squeeze(dim=0).cpu()
        attn_weights.append(attn_weight)
    attn_weights = torch.stack(attn_weights)
    return attn_weights


@torch.no_grad()
def get_dequant_values(
        args: Namespace, quantizer: QuantizerBase, predictor: nn.Module,
        predictor_inputs: Sequence[torch.Tensor], values: Sequence[torch.Tensor]):
    """Return a list of reconstructed (quantized-dequantized with predictor) tensors; values can be K or V"""
    assert len(predictor_inputs) == len(values)
    values_dequantized = []
    for i in trange(len(predictor_inputs), desc='get_dequant_values', leave=False):
        predictor_inputs_i = predictor_inputs[i].to(args.devices[0], non_blocking=True)
        values_i = values[i].to(args.devices[0], non_blocking=True)
        if predictor is None:
            values_pred_i = torch.zeros_like(values_i, device=values_i.device)
        else:
            values_pred_i = predictor(predictor_inputs_i)
        values_delta = values_i - values_pred_i
        values_delta_dequant_i = quantizer.quantize_dequantize(values_delta.flatten(0, -2)).reshape(values_delta.shape)
        values_dequantized.append(
            (values_delta_dequant_i + values_pred_i).to(values[i].device, non_blocking=True))
    return values_dequantized


@torch.no_grad()
def compute_relative_mse(
        predictor: nn.Module, X: torch.Tensor, Y: torch.Tensor, chunk_size: int = None,
        compute_device: Optional[torch.device] = None, compute_dtype: torch.dtype = None
) -> float:
    """Compute ||predictor(X) - Y||^2 / ||Y||^2 in a memory-efficient manner"""
    if compute_device is None:
        compute_device = next(predictor.parameters()).device

    if chunk_size is None:
        return ((predictor(X) - Y).norm() / Y.norm()).item() ** 2
    else:
        numerator = denominator = 0
        for chunk_start in trange(0, len(X), chunk_size, desc='compute_relative_mse', leave=False):
            xb, yb = [tensor[chunk_start: chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True)
                for tensor in (X, Y)]
            numerator += (predictor(xb) - yb).norm().square().item()
            denominator += yb.norm().square().item()
        return numerator / denominator

def make_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model_name",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--edenn_d",
        type=int,
        help="The grid dimension d for HIGGS.",
    )
    parser.add_argument(
        "--edenn_n",
        type=int,
        help="The grid size n for HIGGS.",
    )
    parser.add_argument(
        "--not_quantize_first_layer",
        action="store_true",
        help="If this flag is set, the first layer will not be quantized.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default=None,
        help="dtype for computing activations",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=8192,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument("--devices",
                        metavar="N",
                        type=str,
                        nargs="+",
                        default=None,
                        help="List of devices")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
             "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--total_nsamples",
        type=int,
        default=256,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--valid_nsamples",
        type=int,
        default=32,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Number of tokens in one chunk.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=1e-3,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--hadamard_groupsize",
        type=int,
        default=1024,
        help="Groupsize of Hadamard transform for HIGGS.",
    )
    parser.add_argument(
        "--predictors_output_path",
        type=str,
        default="./key_value_predictors.pt",
        help="Path to save trained predictors for Key and Values",
    )

    return parser


def main():

    # parse args
    parser = make_arg_parser()

    torch.set_num_threads(min(16, torch.get_num_threads()))

    args = parser.parse_args()
    # infer defaults
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert len(args.devices) == 1, "training-time parallelism is not implemented yet"

    # load model and data
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True,
        use_cache=False
    )
    config = transformers.AutoConfig.from_pretrained(args.model_name)
    print(model.config._attn_implementation)

    data = datautils.get_loaders(
        args.dataset,
        nsamples=args.total_nsamples,
        seed=args.seed,
        model_path=args.model_name,
        seqlen=args.model_seqlen,
        args=args
    )

    common_quantizer_kwargs = dict(
        hadamard_groupsize = args.hadamard_groupsize, 
        channel_size=config.head_dim * config.num_key_value_heads
    )

    quantizer = HiggsQuantizer(
        edenn_d=args.edenn_d, 
        edenn_n=args.edenn_n, 
        **common_quantizer_kwargs
    )
    
    if args.not_quantize_first_layer:
        first_layer_quantizer = None
    else:
        first_layer_quantizer = HiggsQuantizer(
            edenn_d=2, 
            edenn_n=256,
            **common_quantizer_kwargs
    )

    # Calibration: propagate a set of inputs through one layer at a time, train predictors as we go
    layers = modelutils.get_layers(model)

    inps, forward_args = modelutils.get_inps(
        model, data, args.model_seqlen, args.devices, args.offload_activations)

    for k, v in forward_args.items():
        forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

    # this is to make it compatible with transformers 4.48>=
    model.model.rotary_emb.to(args.devices[0])
    forward_args["position_embeddings"] = model.model.rotary_emb.forward(
        inps[0][:1].to(args.devices[0]), torch.arange(0, args.model_seqlen).unsqueeze(0).to(args.devices[0]))
    model.model.rotary_emb.cpu()

    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    old_attn_keys = None
    old_attn_values = None

    key_predictors = {}
    value_predictors = {}

    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        layer_device_original = next(layers[layer_index].parameters()).device
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        layer = layers[layer_index].to(device=args.devices[0], dtype=args.compute_dtype or layer_dtype_original)

        layer.self_attn.k_proj = OutputCatcher(layer.self_attn.k_proj, args.offload_activations)
        layer.self_attn.v_proj = OutputCatcher(layer.self_attn.v_proj, args.offload_activations)
        layer.self_attn.q_proj = OutputCatcher(layer.self_attn.q_proj, args.offload_activations)

        modelutils.update_outs_inplace_(args.devices, layer, inps, outs, **forward_args, compute_mse=False)

        attn_keys = layer.self_attn.k_proj.outputs
        assert all(elem.shape[0] == 1 for elem in attn_keys)
        attn_keys = [elem[0] for elem in attn_keys]

        attn_values = layer.self_attn.v_proj.outputs
        assert all(elem.shape[0] == 1 for elem in attn_values)
        attn_values = [elem[0] for elem in attn_values]

        key_states = torch.stack(layer.self_attn.k_proj.outputs).to(device=args.devices[0])
        query_states = torch.stack(layer.self_attn.q_proj.outputs).to(device=args.devices[0])    

        layer.self_attn.k_proj = layer.self_attn.k_proj.inner
        layer.self_attn.v_proj = layer.self_attn.v_proj.inner
        layer.self_attn.q_proj = layer.self_attn.q_proj.inner
        
        layers[layer_index] = layer.to(device=layer_device_original, dtype=layer_dtype_original)
        
        inps, outs = outs, inps

        if layer_index == 0:
            old_attn_keys = attn_keys
            old_attn_values = attn_values
            if args.not_quantize_first_layer:
                print("Not quantizing first layer")
                continue

        ### training predictor below ###
        key_predictor_inputs = list(old_attn_keys)

        if layer_index == 0:
            key_predictor, mse_train_keys, mse_valid_keys = None, 10000, 10000
        else:
            key_predictor, mse_train_keys, mse_valid_keys = get_predictor(args, key_predictor_inputs, attn_keys)

            prev_layer_keys = torch.stack(key_predictor_inputs).to(device=args.devices[0])
            attention_weights_error(prev_layer_keys, key_states, query_states, model, layer, key_predictor, args)

            del layer
            torch.cuda.empty_cache()

        # attn_keys = get_dequant_values(args, quantizer if layer_index != 0 else first_layer_quantizer, key_predictor, key_predictor_inputs, attn_keys)
        del key_predictor_inputs
        if layer_index != 0:
            key_predictors[layer_index] = key_predictor.cpu()
        train_bits_keys = - math.log(mse_train_keys) / math.log(4)
        valid_bits_keys = - math.log(mse_valid_keys) / math.log(4)
        print(f'{layer_index=}\tPREDICTOR_KEYS   \t| relMSE train: {mse_train_keys:.4f} valid: {mse_valid_keys:.4f} '
              f'| equiv.bits train: {train_bits_keys:.2f} valid: {valid_bits_keys:.2f}')
        value_predictor_inputs = [
            torch.cat([k_i, old_v_i], dim=-1) for k_i, old_v_i in zip(attn_keys, old_attn_values)]
        if layer_index == 0:
            value_predictor, mse_train_values, mse_valid_values = None, 10000,10000
        else:
            value_predictor, mse_train_values, mse_valid_values = get_predictor(args, value_predictor_inputs, attn_values)
        attn_values = get_dequant_values(args, quantizer if layer_index != 0 else first_layer_quantizer, value_predictor, value_predictor_inputs, attn_values)
        if layer_index != 0:
            value_predictors[layer_index] = value_predictor.cpu()
        
        del value_predictor_inputs
        train_bits_values = - math.log(mse_train_values) / math.log(4)
        valid_bits_values = - math.log(mse_valid_values) / math.log(4)
        print(
            f'{layer_index=}\tPREDICTOR_VALUES \t| relMSE train: {mse_train_values:.4f} valid: {mse_valid_values:.4f} '
            f'| equiv.bits train: {train_bits_values:.2f} valid: {valid_bits_values:.2f}')

        old_attn_keys, old_attn_values = attn_keys, attn_values

    torch.save(dict(key_predictors=key_predictors, value_predictors=value_predictors), args.predictors_output_path)
    print("Saved predictors to", args.predictors_output_path)


if __name__ == "__main__":
    main()
