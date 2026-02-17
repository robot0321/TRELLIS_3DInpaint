#!/bin/bash
BASE_PROMPT=$1 #"Trees on the grass"
INPAINT_PROMPT=$2 #"A red roof house"
echo "Base prompt: $BASE_PROMPT"
echo "Inpaint prompt: $INPAINT_PROMPT"
INPAINT_MASK="mask_preset/righthalf.pt"

gpus=(0 1 2 3)
seeds=(0 1 2 3)
inp_seeds=(0 1 2 3)

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    gpu=${gpus[$i]}
    (
        for inp_seed in "${inp_seeds[@]}"; do
            echo "Running on GPU $gpu seed $seed/$inp_seed"
            CUDA_VISIBLE_DEVICES=$gpu python example_inpaint.py \
                --base_prompt "$BASE_PROMPT" \
                --inpaint_prompt "$INPAINT_PROMPT" \
                --inpaint_mask "$INPAINT_MASK" \
                --seed $seed \
                --seed_inpaint $inp_seed \
                --tag "test_draft1"
        done     
    ) &
done
