from omegaconf import OmegaConf
import torch
import tqdm
import transformers
import hydra
from hydra.utils import instantiate
from aquakv.cache_utils import InferenceCache

def test(input_data, model, cache, max_len=None):
    reference_hidden = torch.load("reference_hidden_states.pt")
    print(type(reference_hidden))
    print(len(reference_hidden))
    max_len = max_len or input_data.shape[1]
    lm_logits = torch.zeros(
                    (input_data.shape[0], max_len, model.get_output_embeddings().out_features), 
                    device=input_data.device, 
                    dtype=torch.float
    )
    for i in tqdm.tqdm(range(0, max_len)):
        out = model(input_data[:, i: i + 1], use_cache=True, past_key_values=cache, optimise_aquakv_inference=True)
        lm_logits[:, i: i + 1, ...] = out.logits
        # break
    reference_logits = torch.load("reference_logits.pt")
    # l = 0
    # for layer in range(29):
        # print(l, layer, reference_hidden[layer].shape, reference_hidden[layer].shape, torch.max(torch.abs(reference_hidden[layer][0, l, :] - reference_hidden[layer][0, l, :])))

    for l in range(max_len):
        # print(reference_logits[0, l, :10])
        # print(lm_logits[0, l, :10])
        print(l, torch.max(torch.abs(reference_logits[0, l, :] - lm_logits[0, l, :])), torch.argmax(reference_logits[0, l, :]) == torch.argmax(lm_logits[0, l, :]))
    
    to_break = False
    for j in range(1, 28):
        reference_idx = torch.load(f"reference_idx_{j}.pt")
        reference_scales = torch.load(f"reference_scales_{j}.pt")
        for l in range(max_len):
            print(j, l, torch.any(cache.key_cache[j][0, l].cpu() != reference_idx[l]))
            if torch.any(cache.key_cache[j][0, l].cpu() != reference_idx[l]):
                to_break = True
                print(j, l, " idx ", cache.key_cache[j][0, l], '\n', reference_idx[l])
                print(j, l, " scales ", cache.key_scales[j][0, l], '\n', reference_scales[l].squeeze())
        if to_break:
            break


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
    if "quantizer" in cache_kwargs:
        head_dim = (
            model_config.head_dim if hasattr(model_config, "head_dim") else model_config.hidden_size // model_config.num_attention_heads
        )
        num_key_value_heads = (
            model_config.num_attention_heads
            if getattr(model_config, "num_key_value_heads", None) is None
            else model_config.num_key_value_heads
        )
        channel_size = num_key_value_heads * head_dim
        
        cache_kwargs["quantizer"]["channel_size"] = channel_size
        cache_kwargs["quantizer"]["hadamard_groupsize"] = channel_size

        target_class_name = cache_kwargs["quantizer"]["target"]
        cache_kwargs["quantizer"].pop("target")
        quantizer = instantiate(dict(_target_=target_class_name, **cache_kwargs["quantizer"]))
        cache_kwargs["quantizer"] = quantizer

    cache = instantiate(cache_kwargs)
    
    if hasattr(experiment_config, "test") and experiment_config.test:
        input_data = torch.load("./real_input.pt")
        test(input_data, model, cache, 128)
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