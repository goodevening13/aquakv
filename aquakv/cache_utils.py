"""
Extensions to transformers.cache_utils that enable predictors, quantization, attention_sings, etc
"""
import warnings
from typing import Any, Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from aquakv.quantizers import QuantizerBase


class TreatPrefixSeparately(transformers.cache_utils.Cache):
    def __init__(
            self, prefix_size: int,
            prefix_cache: transformers.cache_utils.Cache,
            suffix_cache: transformers.cache_utils.Cache):
        super().__init__()
        self.prefix_size = prefix_size
        self.prefix_cache, self.suffix_cache = prefix_cache, suffix_cache

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        num_new_tokens = key_states.shape[-2]
        num_added_to_prefix = min(self.prefix_size - self.prefix_cache.get_seq_length(layer_idx), num_new_tokens)

        if num_added_to_prefix == num_new_tokens and num_new_tokens > 0:
            assert self.suffix_cache.get_seq_length(layer_idx) == 0  # writing to prefix only
            return self.prefix_cache.update(key_states, value_states, layer_idx, cache_kwargs)

        else:
            if cache_kwargs and set(cache_kwargs.keys()) != {'sin', 'cos', 'cache_position'}:
                warnings.warn(f"{self.__class__.__name__} was not tested with {cache_kwargs=}")
            prefix_cache_kwargs = {
                k: (v[..., :num_added_to_prefix, :] if v.ndim > 1 else v[:num_added_to_prefix])
                for k, v in cache_kwargs.items()} if cache_kwargs else cache_kwargs
            suffix_cache_kwargs = {
                k: (v[..., num_added_to_prefix:, :] if v.ndim > 1 else v[num_added_to_prefix:])
                for k, v in cache_kwargs.items()} if cache_kwargs else cache_kwargs
            prefix_keys, prefix_values = self.prefix_cache.update(
                key_states[..., :num_added_to_prefix, :], value_states[..., :num_added_to_prefix, :],
                layer_idx, prefix_cache_kwargs)
            suffix_keys, suffix_values = self.suffix_cache.update(
                key_states[..., num_added_to_prefix:, :], value_states[..., num_added_to_prefix:, :],
                layer_idx, suffix_cache_kwargs)

            return torch.cat([prefix_keys, suffix_keys], dim=-2), torch.cat([prefix_values, suffix_values], dim=-2)

    def get_seq_length(self, **kwargs) -> int:
        return self.prefix_cache.get_seq_length(**kwargs) + self.suffix_cache.get_seq_length(**kwargs)


class PredictorHiggsCache(transformers.cache_utils.Cache):
    """
    A cache with layer-wise predictors and residual HIGGS quantization;
    quantizes data after processing at least :buffer_size: tokens (or more with batch encoding)
    :param config: model config, used to determine the number of attention layers to cache
    :param min_buffer_size: keep last KVs un-quantized until this many of them (or more) are accumulated
    :param make_quantized_cache: return an empty transformers Cache instance that will be used to store each buffer
    :param save_dequantized_values: if True, save de-quantized keys/values to avoid repeated de-quantization
        This option is meant for faster eval / prototyping when de-quantization code is inefficient
    """

    def __init__(self, *, config: transformers.PretrainedConfig, make_quantized_cache: callable, min_buffer_size: int,
                 save_dequantized_values: bool = False):
        super().__init__()
        self.make_quantized_cache = make_quantized_cache
        self.save_dequantized_values = save_dequantized_values
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.buffer_size = min_buffer_size
        self.buffer_cache = transformers.cache_utils.DynamicCache()
        self.buffer_kwargs = []
        self.combined_buffer_kwargs = None

        self.quantized_caches: List[SingleChunkQuantizedCacheWithPredictors] = []
        self.latest_quantized_cache = self.make_quantized_cache()

        self.next_layer_idx = 0
        self.compressing = False

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx in (self.next_layer_idx, 0), (layer_idx, self.next_layer_idx, 0)
        if layer_idx == 0:
            self.buffer_kwargs.append(cache_kwargs)
            assert set(cache_kwargs.keys()) == {'sin', 'cos', 'cache_position'}
        empty = key_states[..., :0, :]
        key_buffer, value_buffer = self.buffer_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        empty_kwargs = {k: (v[..., :0, :] if v.ndim > 1 else v[:0])
                        for k, v in cache_kwargs.items()} if cache_kwargs else cache_kwargs
        dequantized_key_chunks, dequantized_value_chunks = zip(
            *[cache.update(empty, empty, layer_idx, empty_kwargs) for cache in self.quantized_caches
              ] + [(key_buffer, value_buffer)])
        combined_key_states = torch.cat(dequantized_key_chunks, dim=-2)
        combined_value_states = torch.cat(dequantized_value_chunks, dim=-2)

        if not self.compressing and key_buffer.shape[-2] >= self.buffer_size:
            self.compressing = True
            self.combined_buffer_kwargs = self.combine_buffer_kwargs()
            self.buffer_kwargs = []

        if self.compressing:
            self.latest_quantized_cache.update(
                key_buffer, value_buffer, layer_idx, self.combined_buffer_kwargs
            )
            self.buffer_cache.key_cache[layer_idx] = empty.clone()
            self.buffer_cache.value_cache[layer_idx] = empty.clone()

        if self.compressing and layer_idx == self.num_layers - 1:  # compression done
            cache_to_save = self.latest_quantized_cache
            if self.save_dequantized_values:
                cache_to_save = FrozenCache(cache_to_save, config=self.config)
            self.quantized_caches.append(cache_to_save)

            self.latest_quantized_cache = self.make_quantized_cache()
            self.buffer_cache = transformers.cache_utils.DynamicCache()
            self.combined_buffer_kwargs = None
            self.compressing = False
        self.next_layer_idx = layer_idx + 1
        return combined_key_states, combined_value_states

    def combine_buffer_kwargs(self):
        assert len(self.buffer_kwargs) > 0
        return dict(cos=torch.cat([kw['cos'] for kw in self.buffer_kwargs], dim=-2),
                    sin=torch.cat([kw['sin'] for kw in self.buffer_kwargs], dim=-2),
                    cache_position=torch.cat([kw['cache_position'] for kw in self.buffer_kwargs], dim=0))

    def get_seq_length(self, *args, **kwargs) -> int:
        return self.buffer_cache.get_seq_length(*args, **kwargs) + sum(
            quantized_cache.get_seq_length(*args, **kwargs) for quantized_cache in self.quantized_caches)


class SingleChunkQuantizedCacheWithPredictors(transformers.cache_utils.Cache):
    """A **write-once** cache that uses cumulative predictors; assumes that inputs are pre-grouped"""

    def __init__(self, *, quantizer: QuantizerBase, first_layer_quantizer: QuantizerBase = None,
                 key_predictors: Optional[Dict[int, nn.Module]] = None,
                 value_predictors: Optional[Dict[int, nn.Module]] = None,
                 move_predictors_to_devices: bool = True):
        super().__init__()
        self.quantizer, self.key_predictors, self.value_predictors = quantizer, key_predictors, value_predictors
        self.first_layer_quantizer = first_layer_quantizer
        self.key_states_cache, self.value_states_cache, self.device_map = dict(), dict(), dict()
        self.previous_key_reconstruction = self.previous_value_reconstruction = None
        self.next_layer_idx = 0
        self.seq_length = 0
        self.cos = self.sin = None
        self.head_dim = None
        self.move_predictors_to_devices = move_predictors_to_devices

    def predict_next_key_states(self) -> torch.Tensor:
        if self.key_predictors is not None:
            if self.move_predictors_to_devices:
                predictor_device = next(self.key_predictors[self.next_layer_idx].parameters()).device
                if predictor_device != self.previous_key_reconstruction.device:
                    self.key_predictors[self.next_layer_idx].to(self.previous_key_reconstruction.device)
            return self.key_predictors[self.next_layer_idx](self.previous_key_reconstruction)
        else:
            return torch.zeros_like(self.previous_key_reconstruction)

    def predict_next_value_states(self, reconstructed_key_states: torch.Tensor) -> torch.Tensor:
        if self.value_predictors is not None:
            value_predictor_inputs = torch.cat(
                [reconstructed_key_states.to(self.previous_value_reconstruction.device),
                 self.previous_value_reconstruction], dim=-1)
            if self.move_predictors_to_devices:
                predictor_device = next(self.value_predictors[self.next_layer_idx].parameters()).device
                if predictor_device != value_predictor_inputs.device:
                    self.value_predictors[self.next_layer_idx].to(value_predictor_inputs.device)
            return self.value_predictors[self.next_layer_idx](value_predictor_inputs)
        else:
            return torch.zeros_like(self.previous_value_reconstruction)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        assert layer_idx == 0
        return self.key_states_cache[0].shape[-2] if self.key_states_cache else 0

    @torch.no_grad()
    def update(self,
               key_states: Optional[torch.Tensor],
               value_states: Optional[torch.Tensor],
               layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx in (self.next_layer_idx, 0), (layer_idx, self.next_layer_idx, 0)
        assert (key_states is None and value_states is None) or (key_states.shape == value_states.shape)
        saving_new_entries = key_states is not None and key_states.numel() != 0
        assert saving_new_entries == (layer_idx not in self.key_states_cache), "can only write once per layer"
        assert key_states.device == value_states.device and key_states.dtype == value_states.dtype

        if saving_new_entries:  # write mode
            device, dtype = key_states.device, key_states.dtype
            key_states_original, value_states_original = key_states, value_states
            assert 'sin' in cache_kwargs and 'cos' in cache_kwargs
            if self.cos is None:  # save the (identical) sin/cos for future reuse
                self.cos, self.sin = cache_kwargs['cos'], cache_kwargs['sin']

            if self.head_dim is None:
                self.head_dim = key_states.shape[-1]
            # undo rotation using cos(-alpha) = cos(alpha) and sin(-alpha) = -sin(alpha)
            key_states = apply_rotary_to_keys(key_states, cos=self.cos.to(device), sin=-self.sin.to(device))

            # v-- from [batch, num_heads, seq_length, head_dim] to [batch, seq_length, hidden_size]
            key_states, value_states = map(combine_heads, (key_states, value_states))

            if layer_idx == 0:
                if self.first_layer_quantizer:
                    ### hacking to not debug
                    self.quantized_first_layer_k_cache = self.first_layer_quantizer.quantize(
                        (key_states).flatten(0, -2))
                    reconstructed_key_states = self.first_layer_quantizer.dequantize(
                        self.quantized_first_layer_k_cache).view_as(key_states).to(dtype=dtype, device=device)
                    self.key_states_cache[0] = reconstructed_key_states

                    self.quantized_first_layer_v_cache = self.first_layer_quantizer.quantize(
                        (value_states).flatten(0, -2))
                    reconstructed_value_states = self.first_layer_quantizer.dequantize(
                        self.quantized_first_layer_v_cache).view_as(value_states).to(dtype=dtype, device=device)
                    self.value_states_cache[0] = reconstructed_value_states
                else:
                    reconstructed_key_states = self.key_states_cache[0] = key_states
                    reconstructed_value_states = self.value_states_cache[0] = value_states
            else:
                predicted_key_states = self.predict_next_key_states().to(device)
                self.key_states_cache[layer_idx] = self.quantizer.quantize(
                    (key_states - predicted_key_states).flatten(0, -2))
                reconstructed_key_states = predicted_key_states + self.quantizer.dequantize(
                    self.key_states_cache[layer_idx]).view_as(key_states).to(dtype=dtype, device=device)
                predicted_value_states = self.predict_next_value_states(reconstructed_key_states).to(device)
                self.value_states_cache[layer_idx] = self.quantizer.quantize(
                    (value_states - predicted_value_states).flatten(0, -2))
                reconstructed_value_states = predicted_value_states + self.quantizer.dequantize(
                    self.value_states_cache[layer_idx]).view_as(value_states).to(dtype=dtype, device=device)

            # return original data since it's available, avoid quantization errors for that one step
            result_key, result_value = key_states_original, value_states_original
        else:  # read mode
            if layer_idx == 0:
                reconstructed_key_states = self.key_states_cache[0]
                reconstructed_value_states = self.value_states_cache[0]
                device, dtype = reconstructed_key_states.device, reconstructed_key_states.dtype
            else:
                dtype = key_states.dtype
                device = next(iter(self.key_states_cache[self.next_layer_idx].state_dict().values())).device # self.quantizer.device
                reconstructed_key_states = self.predict_next_key_states().to(device)
                reconstructed_key_states += self.quantizer.dequantize(
                    self.key_states_cache[layer_idx]).view_as(reconstructed_key_states).to(dtype=dtype, device=device)

                reconstructed_value_states = self.predict_next_value_states(reconstructed_key_states).to(device)
                reconstructed_value_states += self.quantizer.dequantize(self.value_states_cache[layer_idx]).view_as(
                    reconstructed_value_states).to(dtype=dtype, device=device)

            # apply rotary embedding again
            assert self.sin is not None and self.cos is not None and self.head_dim is not None
            result_key_without_rotary = split_heads(reconstructed_key_states, self.head_dim)
            result_key = apply_rotary_to_keys(
                result_key_without_rotary, cos=self.cos.to(device), sin=self.sin.to(device))
            result_value = split_heads(reconstructed_value_states, self.head_dim)

        self.next_layer_idx = layer_idx + 1
        self.previous_key_reconstruction = reconstructed_key_states
        self.previous_value_reconstruction = reconstructed_value_states
        return result_key, result_value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_seq_length()})"


def apply_rotary_to_keys(key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    _, key_states = apply_rotary_pos_emb(
        q=key_states[..., :1, :], k=key_states, cos=cos, sin=sin)
    return key_states


def split_heads(key_states: torch.Tensor, head_dim: int) -> torch.Tensor:
    assert key_states.ndim == 3  # [batch, length, hidden_dim] -> [batch, num_heads, length, head_dim]
    return key_states.reshape(*key_states.shape[:2], -1, head_dim).transpose(1, 2)


def combine_heads(key_states: torch.Tensor) -> torch.Tensor:
    assert key_states.ndim == 4  # [batch, num_heads, length, head_dim] -> [batch, length, hidden_dim]
    return key_states.transpose(1, 2).flatten(-2)


class FrozenCache(transformers.cache_utils.DynamicCache):
    def __init__(self, cache: SingleChunkQuantizedCacheWithPredictors, config: transformers.PretrainedConfig):
        super().__init__()
        batch_size = cache.key_states_cache[0].shape[0]
        device, dtype = cache.key_states_cache[0].device, cache.key_states_cache[0].dtype
        cache_length = cache.get_seq_length()
        cache_position = torch.arange(cache_length, device=device)
        past_key_values = get_past_key_values(
            cache=cache, config=config, batch_size=batch_size, device=device, dtype=dtype)
        for layer_idx in range(len(past_key_values)):
            super().update(*past_key_values[layer_idx], layer_idx=layer_idx,
                           cache_kwargs=dict(cache_position=cache_position))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert key_states.numel() == value_states.numel() == 0
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


@torch.no_grad()
def get_past_key_values(
        cache: transformers.cache_utils.Cache, config: transformers.PretrainedConfig,
        batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    empty = torch.zeros(
        batch_size, config.num_key_value_heads, 0, config.head_dim, device=device, dtype=dtype)
    empty_rotary_coeffs = torch.zeros(
        batch_size, config.num_key_value_heads, 0, config.head_dim, device=device, dtype=dtype)

    past_key_values = []
    for layer_idx in range(config.num_hidden_layers):
        past_key_values.append(cache.update(
            empty, empty, layer_idx=layer_idx, cache_kwargs=dict(
                cos=empty_rotary_coeffs, sin=empty_rotary_coeffs, cache_position=torch.arange(0, device=device))))
    return past_key_values
