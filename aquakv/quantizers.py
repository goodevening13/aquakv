"""
Base vector quantizer class to be used for training and inference with KV cache predictors and its instances (e.g HIGGS)
"""
import math
from typing import TypeVar, Union
import torch
from fast_hadamard_transform import hadamard_transform
from .edenn import get_grid, get_grid_norms_squared, higgs_quantize_dequantize, pad_to_block, GRIDS
from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight
from collections import namedtuple

class QuantizerBase:
    QuantizedState = TypeVar('QuantizedState')

    def quantize(self, x: torch.Tensor) -> QuantizedState: ...

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor: ...

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)

QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])

class HiggsQuantizer(QuantizerBase):
    def __init__(self, hadamard_groupsize: int, codeword_dim: int, n_codewords: int, device: Union[str, torch.device], dtype, channel_size: int = 1024, chunk_size: int = 64) -> None:
        """
        chunk_size is used to avoid memory demanding matmul and split the input into chunk of size chunk_size to perform multiple smaller matmuls
        """
        super().__init__()
        self.hadamard_groupsize = hadamard_groupsize
        self.channel_size = channel_size
        self.grid = get_grid(codeword_dim, n_codewords, device).T.to(dtype=dtype) # grid of shape [codeword_dim, n_codewords]
        self.grid_norm = get_grid_norms_squared(codeword_dim, n_codewords, device).to(dtype=dtype)
        self.d = codeword_dim
        self.n = n_codewords
        self.chunk_size = chunk_size
        self.hadamard_scale = 1 / hadamard_groupsize
        self.device = device

    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """
        x.shape - [B, C]
        """
        batch_size = x.shape[0]
        x = pad_to_block(x, [1], self.hadamard_groupsize)
        mult = x.shape[1] // self.hadamard_groupsize
        x = x.reshape(-1, mult, self.hadamard_groupsize)
        scales = torch.linalg.norm(x, axis=-1) # [B, mult]
        x = hadamard_transform(x) / scales[:, :, None]

        x = pad_to_block(x, [2], self.d).reshape(batch_size, mult, -1, self.d)

        result_idx = torch.empty((batch_size, mult, x.shape[2]), dtype=torch.uint8)
        for i, chunk in enumerate(torch.split(x, self.chunk_size, dim=0)):
            chunk_idx = torch.argmax(2 * chunk @ self.grid - self.grid_norm, dim=-1) # [B, mult, pad(pad(C)) / mult / d] # .flatten(start_dim=1)
            result_idx[i * self.chunk_size: (i + 1) * self.chunk_size] = chunk_idx

        return QuantizedTensor(
            result_idx,
            scales
        )

    def dequantize(self, quantized: QuantizedTensor) -> torch.Tensor:
        """
        quantized.idx shape is [B, padded_d(padded_had(C)) // d]
        quantized.scale shape is [B, padded_had(C) // hadamard_groupsize]
        """
        idx = quantized.idx
        scales = quantized.scales
        x = self.grid.T[idx.int()].flatten(start_dim=2)  # [b, mult, C / mult / d, d] -> [b, mult, C / mult]

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
