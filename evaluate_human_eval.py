import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm


from typing import Optional, Callable

import torch
import transformers
from torch import nn as nn
from tqdm import tqdm
from transformers import DynamicCache

from aquakv.quantizers import HiggsQuantizer
from aquakv.cache_utils import TreatPrefixSeparately, PredictorHiggsCache, SingleChunkQuantizedCacheWithPredictors
from functools import partial
from datasets import load_dataset


@torch.no_grad()
def evaluate_humaneval(
        model: nn.Module, tokenizer, save_name, device: torch.device,
        amp_dtype: Optional[torch.dtype] = None,
        cache_factory: Optional[Callable[[], DynamicCache]] = None,
        num_samples_per_task: int = 2,
        ) -> float:
    
    def generate_one_completion(prompt):
        input_batch = [prompt for _ in range(num_samples_per_task)]
        inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
        input_ids_cutoff = inputs.input_ids.size(dim=1)
        if cache_factory is None:
            cache = DynamicCache()
        else:
            cache = cache_factory()
        out = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, use_cache=True, past_key_values=cache)

        batch_completions = tokenizer.batch_decode(
            [ids[input_ids_cutoff:] for ids in out],
            skip_special_tokens=True,
        )
        return batch_completions
    
    problems = read_problems()

    samples = []
    for task_id in tqdm(problems):
        generated = generate_one_completion(problems[task_id]["prompt"])
        for j in range(num_samples_per_task):
            samples.append(dict(task_id=task_id, completion=generated[j]))

    write_jsonl(save_name, samples)


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
    parser.add_argument(
        "--not_quantize_first_layer",
        action="store_true",
        help="If this flag is set, the first layer will not be quantized.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=13,
        help="Random seed for stability testing",
    )
    parser.add_argument("--prefix_size",
                        type=int,
                        default=4,
                        help="The number of first tokens that will not be quantized, because of attention sink.")
    parser.add_argument("--no_quant", action="store_true", help="Do not quantize.")
    parser.add_argument("--save_name",
                        type=str,
                        default="samples.jsonl")

    return parser


def main():
    parser = make_arg_parser()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading predictors
    key_predictors, value_predictors = None, None
    if args.predictors_input_path:
        key_values_predictors = torch.load(args.predictors_input_path, weights_only=False)
        key_predictors, value_predictors = key_values_predictors["key_predictors"], key_values_predictors["value_predictors"]
        [key_predictors[i].to(device) for i in key_predictors]
        [value_predictors[i].to(device) for i in value_predictors]

    # loading model
    config = transformers.AutoConfig.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True, device_map='auto')
    print(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, config=config, padding_side="left")

    with torch.no_grad():
        if args.no_quant:
            cache_factory = None
        else:
            quantizer = HiggsQuantizer(args.hadamard_groupsize, args.edenn_d, args.edenn_n)
            if args.not_quantize_first_layer:
                first_layer_quantizer = None
            else:
                first_layer_quantizer = HiggsQuantizer(args.hadamard_groupsize, 2, 256)
       
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
                        value_predictors=value_predictors,
                        first_layer_quantizer=first_layer_quantizer
                    )
                )
            )

        evaluate_humaneval(model, tokenizer, save_name=args.save_name, device=device, cache_factory=cache_factory, num_samples_per_task=200)


if __name__ == "__main__":
    main()
