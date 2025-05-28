"""
Base vector quantizer class to be used for training and inference with KV cache predictors and its instances (e.g HIGGS)
"""
from typing import TypeVar
import torch
from fast_hadamard_transform import hadamard_transform
from .edenn import higgs_quantize_dequantize, pad_to_block, HadLinear


class QuantizerBase:
    QuantizedState = TypeVar('QuantizedState')

    def quantize(self, x: torch.Tensor) -> QuantizedState: ...

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor: ...

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)


class HiggsQuantizer(QuantizerBase):
    """
    HIGGS vector quantization. This version is highly inefficient, but convenient for prototyping.
    :param hadamard_groupsize: perform random hadamard transform to groups of this many vectors
    :param edenn_d: quantization grouop dimension
    :param edenn_n: quantization lattice size
    """
    def __init__(self, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
        super().__init__()
        self.hadamard_groupsize, self.edenn_d, self.edenn_n = hadamard_groupsize, edenn_d, edenn_n

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        return quantize_linear_weight_higgs(x, self.hadamard_groupsize, self.edenn_d, self.edenn_n)

    @torch.no_grad()
    def dequantize(self, quantized: HadLinear) -> torch.Tensor:
        device = quantized.weight.device if quantized.weight.device.type == 'cuda' else 'cuda:0'
        return quantized(torch.eye(quantized.weight.shape[1], device=device).half()).T.contiguous()

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:  # note: this shortcut is likely useless :D
        output_layer = quantize_linear_weight_higgs(x, self.hadamard_groupsize, self.edenn_d, self.edenn_n)
        device = x.device if x.device.type == 'cuda' else 'cuda:0'
        return output_layer(torch.eye(x.shape[1], device=device).half()
                            ).T.detach().contiguous().clone().to(device=x.device, dtype=x.dtype)


@torch.no_grad()
def quantize_linear_weight_higgs(weight: torch.Tensor, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    """HIGGS quantization code for weights reused from https://arxiv.org/abs/2411.17525"""
    weight = weight.to(dtype=torch.float32, device='cuda' if weight.device.type != 'cuda' else weight.device)
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], hadamard_groupsize)

    # Scale and Hadamard transform
    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.reshape(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]

    # Pad to edenn_d and project
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], mult, -1, edenn_d)

    for i in range(0, weight.shape[0], 64):
        weight[i: i + 64] = higgs_quantize_dequantize(weight[i:i + 64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], mult, -1)

    # Cut the padded values
    weight = weight[..., :hadamard_groupsize]

    # Unscale
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)

    return HadLinear(weight.half(), hadamard_groupsize)


class QuantoQuantizer(QuantizerBase):
    """
    Quanto-based quantizer, defaults to per-token groups
    :param nbits: base quantization bitwidth
    :param q_group_size: fit a separate scale for this many consecutive values
    :param axis: axis = 0 is per-token compression; axis = -1 is per-channel
    """

    def __init__(self, nbits: int, q_group_size: int = 64, axis: int = 0):
        from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight
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
