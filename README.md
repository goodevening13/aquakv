# AQUA-KV

[![arXiv](https://img.shields.io/badge/arXiv-2501.19392-b31b1b.svg)](https://arxiv.org/abs/2501.19392)

This is the supplementary code for `Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models`.

The current code version is designed for reproducing and prototyping. Efficient inference kernels are TBU.

# Installation

Install packages from requirements.txt:

`pip install -r requirements.txt`

While we do not make guarantees about the correctness of this code with alternative versions, the code should run
fine with torch 2.5-2.6, transformers 4.46-4.48. The rest of the dependencies are required for LongBench and Quanto.


# Training and evaluation

Below is a minimal example for training and evaluating

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# higgs and buffering
EDENN_D=2
EDENN_N=16
HADAMARD_GROUPSIZE=1024
RECENT_BUFFER_SIZE=128
PREFIX_SIZE=4

# model
MODEL_PATH=unsloth/Llama-3.2-3B
MODEL_SEQLEN=8192


# calibration
CALIBRATION_DATASET=pajama
TOTAL_NSAMPLES=256
VALIDATION_NSAMPLES=32
PREDICTORS_SAVE_PATH=./my_little_predictor.pt

# LongBench
LONGBENCH_MODEL_NAME=llama-3.2-3B
LONGBENCH_OUT_PATH=./test_predictions
LONGBENCH_DATASETS=samsum,2wikimqa,trec,hotpotqa,multi_news,triviaqa,qmsum,passage_count,multifieldqa_en,musique,qasper,passage_retrieval_en,narrativeqa,gov_report
# ^-- note: here is the list of all benchmark tasks in English, some of them are memory-heavy, e.g. narrativeqa;

python train_predictors.py --model_name $MODEL_PATH --model_seqlen $MODEL_SEQLEN --predictors_output_path $PREDICTORS_SAVE_PATH \
  --dataset $CALIBRATION_DATASET --total_nsamples $TOTAL_NSAMPLES --valid_nsamples $VALIDATION_NSAMPLES \
  --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE --offload_activations
# ^-- note: train_predictors currently ignores $PREFIX_SIZE (attention sinks) and RECENT_BUFFER_SIZE (buffering) during calibration;

python evaluate_perplexity.py --model_name $MODEL_PATH --model_seqlen $MODEL_SEQLEN --predictors_input_path $PREDICTORS_SAVE_PATH \
  --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE \
  --prefix_size $PREFIX_SIZE --recent_buffer_size $RECENT_BUFFER_SIZE
 
python evaluate_longbench.py --model $LONGBENCH_MODEL_NAME --predictors_input_path=$PREDICTORS_SAVE_PATH \
 --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE \
 --prefix_size $PREFIX_SIZE --recent_buffer_size $RECENT_BUFFER_SIZE \
 --datasets=$LONGBENCH_DATASETS --out_path=$LONGBENCH_OUT_PATH --quantize
 
```


# Grid Choise

Code supports several grids for HIGGS quantization:

- Fast GPU supproted grids: EDENN_D = 2, EDENN_N = 16, 64, 256 for quantization in 2, 3, 4 bits correspondingly

- Grids without fast GPU support, but with slightly better quality: 
 * EDENN_D = 4, EDENN_N = 256
 * EDENN_D = 8, EDENN_N = 65536
