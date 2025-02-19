"""
HIGGS vector quantization utilities from https://gist.githubusercontent.com/galqiwi/d8fdeb2c6603ad3e54d72a0801416ad3/raw/3eb0762bebe0e95df5ad17b2f3fa9f4ef3a241ac/edenn.py
"""
import functools
import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform

import pathlib

grids_folder = pathlib.Path(__file__).parent.resolve() / "grids"

GRIDS = {
}
# Read files in the folder and read grids in the EDEN{DIM}_{SIZE}.pt format
for file in grids_folder.iterdir():
    print(f"DEBUGPRINT: reading {file}")
    if file.suffix == ".pt":
        try:
            if file.name.startswith("EDEN"):
                dim, size = map(int, file.stem[4:].split('-'))
            elif file.name.startswith("QUIPSHARP"):
                dim, size = map(int, file.stem[9:].split('-'))
            else:
                raise ValueError("Could not parse grid file name")
        except ValueError:
            print(f"DEBUGPRINT: {file} failed to parse")
            continue
        GRIDS[dim] = GRIDS.get(dim, {})
        if size in GRIDS[dim]:
            warnings.warn(f"Got multiple grids for {dim=} {size=}, overriding with {file}")
        GRIDS[dim][size] = torch.load(file, map_location='cpu').to(torch.float32)
        print(f"DEBUGPRINT: read {dim=} {size=} from {file}")

GRID_NORMS = {k1: {k2: torch.linalg.norm(GRIDS[k1][k2], dim=1) ** 2 for k2 in v1.keys()} for k1, v1 in GRIDS.items()}


@functools.lru_cache()
def get_grid(dim: int, size: int, device: torch.device) -> torch.Tensor:
    return GRIDS[dim][size].to(device)


@functools.lru_cache()
def get_grid_norms_squared(dim: int, size: int, device: torch.device) -> torch.Tensor:
    return torch.linalg.norm(get_grid(dim, size, device), dim=1).square()


def entropy(idx):
    _, counts = torch.unique(idx, return_counts=True)
    counts = counts.to(torch.float)
    return -torch.sum(counts / len(idx) * torch.log2(counts / len(idx))).item()


def higgs_quantize(x, dim, size):
    assert size <= 256
    return torch.argmax(2 * x @ get_grid(dim, size, x.device).T - get_grid_norms_squared(dim, size, x.device), dim=-1).to(torch.uint8)


def higgs_quantize_dequantize(x, dim, size):
    idx = torch.argmax(2 * x @ get_grid(dim, size, x.device).T - get_grid_norms_squared(dim, size, x.device), dim=-1)
    return get_grid(dim, size, x.device)[idx]


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta

    return F.pad(tensor, pad_dims, "constant", value)


class HadLinear(nn.Module):
    def __init__(self, weight, had_block_size=1024):
        super().__init__()
        self.register_buffer('had_block_size', torch.tensor(0))
        self.had_block_size = torch.tensor(had_block_size)
        self.weight = nn.Parameter(weight / math.sqrt(had_block_size))

    def forward(self, input):
        input = pad_to_block(input, [-1], self.had_block_size)
        mult = input.shape[-1] // self.had_block_size
        input = input.reshape(input.shape[:-1] + (mult, self.had_block_size))
        input = hadamard_transform(input, scale=1 / math.sqrt(self.had_block_size))
        input = input.reshape(input.shape[:-2] + (mult * self.had_block_size,))
        return F.linear(input, self.weight)
