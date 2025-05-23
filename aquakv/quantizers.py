"""
Base vector quantizer class to be used for training and inference with KV cache predictors and its instances (e.g HIGGS)
"""
from typing import TypeVar
import torch
from fast_hadamard_transform import hadamard_transform
from .edenn import get_grid, get_grid_norms_squared, pad_to_block, GRIDS
from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight
from collections import namedtuple
from functools import partial

class QuantizerBase:
    QuantizedState = TypeVar('QuantizedState')

    def quantize(self, x: torch.Tensor) -> QuantizedState: ...

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor: ...

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)

QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])

class HiggsQuantizer(QuantizerBase):
    def __init__(self, hadamard_groupsize: int, edenn_d: int, edenn_n: int, channel_size: int = 1024, chunk_size: int = 64) -> None:
        """
        HIGGS vector quantization.
        :param hadamard_groupsize: perform random hadamard transform to groups of this many vectors
        :param edenn_d: quantization grouop dimension
        :param edenn_n: quantization lattice size
        :param channel_size: channel size of keys and values, used to trim padding 
        :param chunk_size: chunk size is used to avoid memory demanding matmul and split the input into chunk to perform multiple smaller matmuls
        """
        super().__init__()
        self.hadamard_groupsize = hadamard_groupsize
        self.channel_size = channel_size
        self.grid = partial(get_grid, dim=edenn_d, size=edenn_n) # grid of shape [edenn_d, edenn_n]
        self.grid_norm = partial(get_grid_norms_squared, dim=edenn_d, size=edenn_n)
        self.edenn_d = edenn_d
        self.chunk_size = chunk_size
        self.hadamard_scale = 1 / hadamard_groupsize

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """
        x.shape - [B, C]
        """
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        assert channel_size == self.channel_size, "channel size from __init__ does not match the channel size of quantize argument. Make sure you create HiggsQuantizer with correct channel size"
        device = x.device
        x = x.to(dtype=torch.float32)
        x = pad_to_block(x, [1], self.hadamard_groupsize)
        mult = x.shape[1] // self.hadamard_groupsize
        x = x.reshape(-1, mult, self.hadamard_groupsize)
        scales = torch.linalg.norm(x, axis=-1) # [B, mult]
        x = hadamard_transform(x) / scales[:, :, None]

        x = pad_to_block(x, [2], self.edenn_d).reshape(batch_size, mult, -1, self.edenn_d)

        result_idx = torch.empty((batch_size, mult, x.shape[2]), dtype=torch.uint8)
        for i, chunk in enumerate(torch.split(x, self.chunk_size, dim=0)):
            chunk_idx = torch.argmax(2 * chunk @ self.grid(device=device).T - self.grid_norm(device=device), dim=-1) # [B, mult, pad(pad(C)) / mult / d]
            result_idx[i * self.chunk_size: (i + 1) * self.chunk_size] = chunk_idx

        return QuantizedTensor(
            result_idx,
            scales
        )

    @torch.no_grad()
    def dequantize(self, quantized: QuantizedTensor) -> torch.Tensor:
        """
        quantized.idx shape is [B, padded_d(padded_had(C)) // d]
        quantized.scale shape is [B, padded_had(C) // hadamard_groupsize]
        """
        idx = quantized.idx
        scales = quantized.scales
        device = scales.device
        x = self.grid(device=device)[idx.int()].flatten(start_dim=2)  # [b, mult, C / mult / d, d] -> [b, mult, C / mult]

        # Cut the padded values
        x = x[..., :self.hadamard_groupsize]

        x = (x * scales.unsqueeze(dim=2)).half()  # [b, mult, C / mult] * [b, mult, 1]
        
        x = hadamard_transform(x, scale=self.hadamard_scale).flatten(start_dim=1)  # [b, mult, C / mult] => [b, C]
        
        return x[:, :self.channel_size]


class QuantoQuantizer(QuantizerBase):
    """
    Quanto-based quantizer, defaults to per-token groups
    :param nbits: base quantization bitwidth
    :param q_group_size: fit a separate scale for this many consecutive values
    :param axis: axis = 0 is per-token compression; axis = -1 is per-channel
    """

    def __init__(self, nbits: int, q_group_size: int = 64, axis: int = 0):
        super().__init__()
        self.nbits, self.axis, self.q_group_size = nbits, axis, q_group_size
        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def quantize(self, tensor):
        scale, zeropoint = self.optimizer(tensor, self.qtype, self.axis, self.q_group_size)
        qtensor = quantize_weight(tensor, self.qtype, self.axis, scale, zeropoint, self.q_group_size)
        return qtensor

    def dequantize(self, qtensor):
        return qtensor.dequantize()
