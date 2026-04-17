#!/bin/bash
GPU=$1
BASE_PROMPT="Trees on the grass"
OPTINIT_PROMPT="A red roof house"
# OPTINIT_PROMPT="A blue car on the road"
OPTINIT_MASK="mask_preset/fronthalf.pt"
echo "Running optinit with GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU python example_optinit.py \
    --base_prompt "$BASE_PROMPT" \
    --optinit_prompt "$OPTINIT_PROMPT" \
    --optinit_mask "$OPTINIT_MASK" \
    --seed 3 \
    --steps 12 \
    --seed_optinit 0 \
    --steps_optinit 12 \
    --ss_lr 5.0 \
    --cfg_ss_optinit 7.5 \
    --cfg_slat_optinit 7.5 \
    --ss_max_iter 15 \
    --slat_max_iter 15 \
    --tag "eval_front" \
    --verbose

