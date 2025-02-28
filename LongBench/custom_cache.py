import torch.nn as nn
import transformers
from typing import Optional, Dict
from aquakv.quantizers import HiggsQuantizer
from aquakv.cache_utils import TreatPrefixSeparately, PredictorHiggsCache, SingleChunkQuantizedCacheWithPredictors
from functools import partial


def get_aqua_cache(device, hadamard_groupsize: int, edenn_n: int, edenn_d: int,
                   recent_buffer_size: int, config: transformers.PretrainedConfig,
                   prefix_size: int = 4,
                   key_predictors: Optional[Dict[int, nn.Module]] = None,
                   value_predictors: Optional[Dict[int, nn.Module]] = None,
                   quantizer_type: str = "higgs",
                   not_quantize_first_layer: bool = False):


    # transfering predictors on to correct device
    if key_predictors:
        # maybe need to clone if several devices acceptable
        for i in key_predictors:
            key_predictors[i].to(device)
    if value_predictors:
        # maybe need to clone if several devices acceptable
        for i in value_predictors:
            value_predictors[i].to(device)

    # creating higgs quantizer
    if quantizer_type == "higgs":
        quantizer = HiggsQuantizer(hadamard_groupsize=hadamard_groupsize,
                                   edenn_d=edenn_d, edenn_n=edenn_n)
    else:
        # for the future
        raise NotImplementedError
    if not_quantize_first_layer:
        first_layer_quantizer = None
    else:
        first_layer_quantizer = HiggsQuantizer(hadamard_groupsize, 2, 256)
       
    # creating cache with predictors
    cache = TreatPrefixSeparately(prefix_size=prefix_size,
                                  prefix_cache=transformers.DynamicCache(),
                                  suffix_cache=PredictorHiggsCache(
                                                 config=config, min_buffer_size=recent_buffer_size,
                                                 save_dequantized_values=True,
                                                 make_quantized_cache=partial(
                                                   SingleChunkQuantizedCacheWithPredictors,
                                                   quantizer=quantizer,
                                                   key_predictors=key_predictors,
                                                   value_predictors=value_predictors,
                                                   first_layer_quantizer = first_layer_quantizer
                                                 )
                                            ))
    return cache
