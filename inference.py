from functools import partial
import time
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import tqdm
import transformers
import hydra
from hydra.utils import instantiate
from aquakv.cache_utils import InferenceCache, PredictorHiggsCache, SingleChunkQuantizedCacheWithPredictors, TreatPrefixSeparately
from aquakv.quantizers import HiggsQuantizer


@hydra.main(config_path='inference_configs', config_name=None)
@torch.no_grad()
def main(experiment_config):
    model_config = transformers.AutoConfig.from_pretrained(experiment_config.model_name)  
    multi_gpu = '70B' in experiment_config.model_name
    if multi_gpu:
        layer_device_map = {i: "cuda:0" for i in range(40)} | {i: "cuda:1" for i in range(40, 80)}    
    else:
        layer_device_map = None
    
    device_map = "auto"
    dtype = torch.bfloat16
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        experiment_config.model_name, 
        config=model_config, 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True, 
        device_map=device_map,
    )
    # print('loaded model')
    # time.sleep(15)
    # print("model sleep ended")
    generation_length = getattr(experiment_config, "generation_length", None) or model_config.max_position_embeddings

    if False:
        cache_kwargs = OmegaConf.to_container(experiment_config.cache, resolve=True)
        
        if not hasattr(model_config, "num_key_value_heads"):
            model_config.num_key_value_heads = model_config.num_attention_heads
        num_key_value_heads = model_config.num_key_value_heads
        # assert num_key_value_heads % world_size == 0
        
        # model_config.num_key_value_heads //= world_size

        cache_kwargs["config"] = model_config
        cache_kwargs["device"] = model.device
        cache_kwargs["dtype"] = model.dtype
        if "quantizer" in cache_kwargs or "first_layer_quantizer" in cache_kwargs:
            head_dim = (
                model_config.head_dim if hasattr(model_config, "head_dim") else model_config.hidden_size // model_config.num_attention_heads
            )
            channel_size = num_key_value_heads * head_dim
            
            for q in ["quantizer", "first_layer_quantizer"]:
                if q in cache_kwargs:
                    cache_kwargs[q]["channel_size"] = channel_size
                    cache_kwargs[q]["hadamard_groupsize"] = channel_size
                    if "device" in cache_kwargs[q]:
                        cache_kwargs[q]["device"] = model.device
                    if "dtype" in cache_kwargs[q]:
                        cache_kwargs[q]["dtype"] = model.dtype

                    target_class_name = cache_kwargs[q]["target"]
                    cache_kwargs[q].pop("target")
                    quantizer = instantiate(dict(_target_=target_class_name, **cache_kwargs[q]))
                    cache_kwargs[q] = quantizer

        cache = instantiate(cache_kwargs, layer_device_map=layer_device_map)
    else:
        quantizer = HiggsQuantizer(
                edenn_d=experiment_config.edenn_d,
                edenn_n=256, 
                hadamard_groupsize=1024,
                channel_size=1024
        )
        first_layer_quantizer = HiggsQuantizer(
            edenn_d=experiment_config.first_layer_quantizer_edenn_d, 
            edenn_n=256, 
            hadamard_groupsize=1024,
            channel_size=1024
        )

        key_predictors, value_predictors = None, None
        key_values_predictors = torch.load("./all_layers_predictors.pt", weights_only=False)
        key_predictors, value_predictors = key_values_predictors["key_predictors"], key_values_predictors["value_predictors"]
        [key_predictors[i].to(model.device) for i in key_predictors]
        [value_predictors[i].to(model.device) for i in value_predictors]


        cache_factory = lambda: TreatPrefixSeparately(
            prefix_size=experiment_config.prefix_size,
            prefix_cache=transformers.DynamicCache(),
            suffix_cache=PredictorHiggsCache(
                config=model.config, 
                min_buffer_size=128,
                save_dequantized_values=False,# True,
                make_quantized_cache=partial(
                    SingleChunkQuantizedCacheWithPredictors,
                    quantizer=quantizer,
                    key_predictors=key_predictors,
                    value_predictors=value_predictors,
                    first_layer_quantizer=first_layer_quantizer
                )
            )
        )

        cache = cache_factory()

    # print('before sleep')
    # time.sleep(15)
    # print("sleep ended")
    assert experiment_config.prefill_size == 4
    prefix_data = torch.randint(0, model_config.vocab_size, (experiment_config.batch_size, experiment_config.prefill_size), device=model.device)
    result = torch.empty((experiment_config.batch_size, generation_length), device=model.device, dtype=int)
    out = model(prefix_data, use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
    result[:, :experiment_config.prefill_size] = prefix_data
    result[:, experiment_config.prefill_size] = torch.argmax(out.logits[:, -1, :], dim=-1)
    torch.cuda.synchronize()
    # print("second sleep after forward")
    # time.sleep(10)
    # print("sleep ended")
    step_size = 1
    times = []
    for i in tqdm.tqdm(range(experiment_config.prefill_size, generation_length, step_size)):
        start_time = time.perf_counter()
        
        out = model(result[:, i: i + step_size], use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        result[:, i + 1: i + 1 + step_size] = torch.argmax(out.logits, dim=-1)
        if i % 1_000 == 0:
            np.save(experiment_config.save_name, np.array(times))

# python my_app.py --config-name alt_config.yaml

if __name__ == "__main__":
    main()

"""
meow 
meeeow
meeeeowww
"""