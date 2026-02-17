#!/bin/bash
GPU=$1
BASE_PROMPT="Trees on the grass"
INPAINT_PROMPT="A red roof house"
INPAINT_MASK="mask_preset/righthalf.pt"
echo "Running inpainting with GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU python example_inpaint.py \
    --base_prompt "$BASE_PROMPT" \
    --inpaint_prompt "$INPAINT_PROMPT" \
    --inpaint_mask "$INPAINT_MASK" \
    --seed 1 \
    --steps 12 \
    --seed_inpaint 2 \
    --steps_inpaint 12 \
    --tag "test_draft1" \
    --verbose


