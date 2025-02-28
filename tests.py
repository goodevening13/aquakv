import torch
import random
import numpy as np

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 


def test_BetterHiggsQuantizer():
    from aquakv.quantizers import HiggsQuantizer, BetterHiggsQuantizer

    hadamard_groupsize = 1024
    edenn_d = 2
    edenn_n = 16
    seed = 24
    device = "cuda:0"
    fix_seed(seed)

    reference_quantizer = HiggsQuantizer(hadamard_groupsize, edenn_d, edenn_n)
    quantizer = BetterHiggsQuantizer(hadamard_groupsize, edenn_d, edenn_n, device)

    input_shape = (8192, 1024)
    a = torch.randn(input_shape, device=device)
    reference_output = reference_quantizer.quantize_dequantize(a)
    output = quantizer.quantize_dequantize(a)

    print(reference_output.shape, output.shape)
    print(torch.max(torch.abs(reference_output - output)), torch.max(torch.abs(reference_output - output) / torch.abs(reference_output)))

if __name__ == "__main__":
    test_BetterHiggsQuantizer()
