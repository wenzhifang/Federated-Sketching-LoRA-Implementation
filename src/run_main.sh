#!/bin/bash

export TOKENIZERS_PARALLELISM=true

rank_type="heavy_tail_low"
num_comm_rounds=600
local_iter_per_round=20
eval_freq=20

accelerate launch --num_processes 4 main.py \
    --client-batch 4 \
    --lora_r 64 \
    --lora_r_min 1 \
    --lora_r_max 32 \
    --clients 50 \
    --num_comm_rounds $num_comm_rounds \
    --eval_freq $eval_freq \
    --local_iter_per_round $local_iter_per_round \
    --rank_type $rank_type

python evaluation_vllm.py --eval_freq $eval_freq --num_comm_rounds $num_comm_rounds --local_iter_per_round $local_iter_per_round --rank_type $rank_type --algorithm "slora64"
