#!/bin/bash
# Run config that got 1.87 bpb on a 3060 Ti (single GPU, small model)
set -euo pipefail

export SEED=1337
export VOCAB_SIZE=1024
export MODEL_DIM=384
export NUM_LAYERS=3
export NUM_HEADS=4
export NUM_KV_HEADS=2
export TRAIN_BATCH_TOKENS=4096
export TRAIN_SEQ_LEN=256
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=1000
export MODEL_KIND=gpt_markov
export MARKOV_LR=0.008
export MARKOV_MIX_INIT=0.06
export MARKOV2_BUCKETS=1024
export MARKOV2_LR=0.004
export MARKOV2_MIX_INIT=0.02
export MARKOV_GATE_THRESHOLD=0.20
export MARKOV2_GATE_THRESHOLD=0.20
export MARKOV_GATE_TEMP=0.05
export MARKOV2_GATE_TEMP=0.05
export MUON_MOMENTUM=0.80

torchrun --standalone --nproc_per_node=1 train_gpt_markov_gated.py
