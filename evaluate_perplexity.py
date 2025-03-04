from typing import Optional, Callable

import torch
import transformers
from torch import nn as nn
from tqdm import tqdm
from transformers import DynamicCache

from aquakv.quantizers import BetterHiggsQuantizer, HiggsQuantizer
from aquakv.cache_utils import TreatPrefixSeparately, PredictorHiggsCache, SingleChunkQuantizedCacheWithPredictors
from functools import partial
from datasets import load_dataset


@torch.no_grad()
def evaluate_perplexity(
        model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device,
        amp_dtype: Optional[torch.dtype] = None, step_size: Optional[int] = None,
        cache_factory: Optional[Callable[[], DynamicCache]] = None
        ) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    if step_size is None:
        step_size = seqlen
    inps = [data[:, start : start + seqlen]
            for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
            ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity")):
        input_ids = input_ids.to(device)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            if cache_factory is None:
                cache = DynamicCache()
            else:
                cache = cache_factory()
            dtype = amp_dtype or next(model.parameters()).dtype
            lm_logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], model.get_output_embeddings().out_features), device=device, dtype=dtype)
            for i in range(0, input_ids.shape[1], step_size):
                out = model(input_ids[:, i: i + step_size], use_cache=True, past_key_values=cache)
                assert out.past_key_values is cache
                lm_logits[:, i: i + step_size, ...] = out.logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()
        else:
            raise RuntimeError

    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()


def make_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model_name",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
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
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=8192,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
             "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of tokens processed in one forward pass for simulated sequential generation.",
    )
    parser.add_argument(
        "--recent_buffer_size",
        type=int,
        default=128,
        help="Accumulate at least this many tokens before quantizing.",
    )
    parser.add_argument(
        "--hadamard_groupsize",
        type=int,
        default=1024,
        help="Groupsize of Hadamard transform for HIGGS.",
    )
    parser.add_argument(
        "--predictors_input_path",
        type=str,
        default=None,
        help="Path to saved trained predictors for Key and Values",
    )
    parser.add_argument("--prefix_size",
                        type=int,
                        default=4,
                        help="The number of first tokens that will not be quantized, because of attention sink.")
    parser.add_argument("--no_quant", action="store_true", help="Do not quantize.")

    return parser


def main():
    parser = make_arg_parser()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading predictors
    key_predictors, value_predictors = None, None
    if args.predictors_input_path:
        key_values_predictors = torch.load(args.predictors_input_path)
        key_predictors, value_predictors = key_values_predictors["key_predictors"], key_values_predictors["value_predictors"]
        [key_predictors[i].to(device) for i in key_predictors]
        [value_predictors[i].to(device) for i in value_predictors]

    # loading model and datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    config = transformers.AutoConfig.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True, device_map='auto')
    print(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, config=config, padding_side="left")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")['input_ids']
    step_size = args.chunk_size

    with torch.no_grad():
        if args.no_quant:
            cache_factory = None
        else:
            quantizer = BetterHiggsQuantizer(
                    args.hadamard_groupsize,
                    args.edenn_d,
                    args.edenn_n,
                    device=device,
                    dtype=config.torch_dtype,
                    channel_size=config.head_dim * config.num_key_value_heads
            )
            cache_factory = lambda: TreatPrefixSeparately(
                prefix_size=args.prefix_size,
                prefix_cache=transformers.DynamicCache(),
                suffix_cache=PredictorHiggsCache(
                    config=model.config, 
                    min_buffer_size=args.recent_buffer_size,
                    save_dequantized_values=True,
                    make_quantized_cache=partial(
                        SingleChunkQuantizedCacheWithPredictors,
                        quantizer=quantizer,
                        key_predictors=key_predictors,
                        value_predictors=value_predictors
                    )
                )
            )

        ppl_quantized = evaluate_perplexity(model, testenc, args.model_seqlen, device=device,
                                            step_size=step_size, cache_factory=cache_factory)

    print(f"WikiText-2 perplexity: {ppl_quantized}\n")


if __name__ == "__main__":
    main()
