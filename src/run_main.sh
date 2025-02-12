#!/bin/bash
export TOKENIZERS_PARALLELISM=true
r=64
ratio=0.125
algorithm="slora_r64_ratio_0125"

accelerate launch --multi_gpu main.py --client-batch 4 --lora_r $r --sketching_ratio $ratio --num_epochs 15 --local_iter_per_round 20 --dataset 'commensense' --algorithm $algorithm

datasets=("boolq" "winogrande" "openbookqa" "ARC-Easy" "ARC-Challenge" "social_i_qa" "piqa" "hellaswag")

for dataset in "${datasets[@]}"; do
    echo "Running evaluation for dataset: $dataset"
    accelerate launch --multi_gpu evaluation_par.py --algorithm $algorithm --dataset "$dataset"
done
