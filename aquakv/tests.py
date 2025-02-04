from functools import partial
from typing import Sequence, Dict

import torch
import transformers
from torch import nn

from aquakv.cache_utils import TreatPrefixSeparately, get_past_key_values, PredictorHiggsCache, \
    SingleChunkQuantizedCacheWithPredictors, dequantize_cache
from aquakv.quantizers import QuantizerBase


def _test_prefix_cache_integrity(model, data, prefix_size: int = 7, chunk_size: int = 32, batch_size: int = 3):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    max_length = data[0].shape[1]
    cache = TreatPrefixSeparately(
        prefix_size=prefix_size,
        prefix_cache=transformers.cache_utils.DynamicCache(),
        suffix_cache=transformers.cache_utils.DynamicCache()
    )

    with torch.no_grad():
        for chunk_start in range(0, max_length, chunk_size):
            model.forward(torch.cat(data[:batch_size]).cuda()[:, chunk_start: chunk_start + chunk_size],
                          past_key_values=cache, use_cache=True)

    past_key_values_ours = get_past_key_values(cache, model.config, batch_size=batch_size, device=device, dtype=dtype)

    cache = transformers.cache_utils.DynamicCache()

    with torch.no_grad():
        for chunk_start in range(0, max_length, chunk_size):
            model.forward(torch.cat(data[:batch_size]).cuda()[:, chunk_start: chunk_start + chunk_size],
                          past_key_values=cache, use_cache=True)

    past_key_values_ref = get_past_key_values(cache, model.config, batch_size=batch_size, device=device, dtype=dtype)

    for layer_idx in range(len(past_key_values_ref)):
        k_ref, v_ref = past_key_values_ref[layer_idx]
        k_ours, v_ours = past_key_values_ours[layer_idx]
        assert torch.allclose(k_ref, k_ours)
        assert torch.allclose(v_ref, v_ours)


def _test_cache_dequantization_lowlevel(
        model: transformers.PreTrainedModel, data: Sequence[torch.Tensor], quantizer: QuantizerBase,
        key_predictors: Dict[int, nn.Module], value_predictors: Dict[int, nn.Module],
        prefix_size: int = 7, chunk_size: int = 32, batch_size: int = 3
):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    max_length = data[0].shape[1]
    cache = TreatPrefixSeparately(
        prefix_size=prefix_size,
        prefix_cache=transformers.cache_utils.DynamicCache(),
        suffix_cache=PredictorHiggsCache(
            config=model.config, min_buffer_size=chunk_size,
            make_quantized_cache=partial(
                SingleChunkQuantizedCacheWithPredictors,
                quantizer=quantizer, key_predictors=key_predictors, value_predictors=value_predictors
    )

        )
    )
    assert isinstance(cache.suffix_cache, PredictorHiggsCache)

    with torch.no_grad():
        for chunk_start in range(0, max_length, chunk_size):
            model.forward(torch.cat(data[:batch_size]).to(device)[:, chunk_start: chunk_start + chunk_size],
                          past_key_values=cache, use_cache=True)

    for buffer_cache in range(len(cache.suffix_cache.quantized_caches)):
        assert isinstance(buffer_cache, SingleChunkQuantizedCacheWithPredictors)
        dequantized_cache = dequantize_cache(buffer_cache, config=model.config)
        kv_before = get_past_key_values(
            buffer_cache, config=model.config, batch_size=batch_size, device=device, dtype=dtype)

        kv_dequantized = get_past_key_values(
            dequantized_cache, config=model.config, batch_size=batch_size, device=device, dtype=dtype)

        for layer_idx in range(model.config.num_hidden_layers):
            k_ref, v_ref = kv_before[layer_idx]
            k_ours, v_ours = kv_dequantized[layer_idx]
            assert torch.allclose(k_ref, k_ours)
            assert torch.allclose(v_ref, v_ours)
