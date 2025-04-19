from omegaconf import OmegaConf
import torch
import tqdm
import transformers
import hydra
from hydra.utils import instantiate
from aquakv.cache_utils import InferenceCache

def test(input_data, model, cache, max_len=None, prefil_size=0):
    max_len = max_len or input_data.shape[1]
    lm_logits = torch.zeros(
                    (input_data.shape[0], max_len, model.get_output_embeddings().out_features), 
                    device=input_data.device, 
                    dtype=torch.float
    )
    if prefil_size > 0:
        out = model(input_data[:, :prefil_size], use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
        lm_logits[:, :prefil_size, ...] = out.logits

    for i in tqdm.tqdm(range(prefil_size, max_len)):
        out = model(input_data[:, i: i + 1], use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
        lm_logits[:, i: i + 1, ...] = out.logits
    reference_logits = torch.load("reference_logits_prefil_136.pt")

    for l in range(max_len):
        print(l, torch.max(torch.abs(reference_logits[0, l, :] - lm_logits[0, l, :])), torch.argmax(reference_logits[0, l, :]) == torch.argmax(lm_logits[0, l, :]))
    

@hydra.main(config_path='inference_configs', config_name=None)
@torch.no_grad()
def main(experiment_config):
    model_config = transformers.AutoConfig.from_pretrained(experiment_config.model_name)  
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        experiment_config.model_name, 
        config=model_config, 
        torch_dtype='float32', # 'auto', 
        low_cpu_mem_usage=True, 
        device_map='auto'
    )

    generation_length = getattr(experiment_config, "generation_length", None) or model_config.max_position_embeddings

    cache_kwargs = OmegaConf.to_container(experiment_config.cache, resolve=True)
    cache_kwargs["config"] = model_config
    cache_kwargs["device"] = model.device
    cache_kwargs["dtype"] = model.dtype
    if "quantizer" in cache_kwargs or "first_layer_quantizer" in cache_kwargs:
        head_dim = (
            model_config.head_dim if hasattr(model_config, "head_dim") else model_config.hidden_size // model_config.num_attention_heads
        )
        num_key_value_heads = (
            model_config.num_attention_heads
            if getattr(model_config, "num_key_value_heads", None) is None
            else model_config.num_key_value_heads
        )
        channel_size = num_key_value_heads * head_dim
        
        for q in ["quantizer", "first_layer_quantizer"]:
            if q in cache_kwargs:
                cache_kwargs[q]["channel_size"] = channel_size
                cache_kwargs[q]["hadamard_groupsize"] = channel_size

                target_class_name = cache_kwargs[q]["target"]
                cache_kwargs[q].pop("target")
                quantizer = instantiate(dict(_target_=target_class_name, **cache_kwargs[q]))
                cache_kwargs[q] = quantizer

    cache = instantiate(cache_kwargs)
    
    if hasattr(experiment_config, "test") and experiment_config.test:
        input_data = torch.load("./real_input.pt")
        prefil_size = experiment_config.prefill_size if hasattr(experiment_config, "prefill_size") and experiment_config.prefill_size is not None else 0
        test(input_data, model, cache, 8 * 128 + 1, prefil_size)
    else:
        prefix_data = torch.randint(0, model_config.vocab_size, (experiment_config.batch_size, experiment_config.prefill_size), device=model.device)
        result = torch.empty((experiment_config.batch_size, generation_length), device=model.device, dtype=int)
        out = model(prefix_data, use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
        result[:, :experiment_config.prefill_size] = prefix_data
        result[:, experiment_config.prefill_size] = torch.argmax(out.logits[:, -1, :], dim=-1)
        torch.cuda.synchronize()

        for i in tqdm.tqdm(range(experiment_config.prefill_size, generation_length)):
            out = model(result[:, i: i + 1], use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
            result[:, i + 1: i + 2] = torch.argmax(out.logits, dim=-1)
        torch.cuda.synchronize()
    
# python my_app.py --config-name alt_config.yaml

if __name__ == "__main__":
    main()