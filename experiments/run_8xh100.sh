#!/bin/bash
# Competition-scale config for 8xH100 — gated Markov + GPT
# Goal: beat 1.2244 baseline bpb
set -euo pipefail

export SEED=1337
export VOCAB_SIZE=1024
export MODEL_DIM=512
export NUM_LAYERS=9
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=20000
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=200
export MODEL_KIND=gpt_markov
export MARKOV_LR=0.008
export MARKOV_MIX_INIT=0.06
export MARKOV2_BUCKETS=2048
export MARKOV2_LR=0.004
export MARKOV2_MIX_INIT=0.02
export MARKOV_GATE_THRESHOLD=0.20
export MARKOV2_GATE_THRESHOLD=0.20
export MARKOV_GATE_TEMP=0.05
export MARKOV2_GATE_TEMP=0.05
export MUON_MOMENTUM=0.85

torchrun --standalone --nproc_per_node=8 train_gpt_markov_gated.py
