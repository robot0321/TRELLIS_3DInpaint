#/bin/bash!

BASE_PROMPT=$1 #"Trees on the grass"
INPAINT_PROMPT=$2 #"A red roof house"
echo "Base prompt: $BASE_PROMPT"
echo "Inpaint prompt: $INPAINT_PROMPT"
logpath="logs/test_draft1"

python make_inpaint_seed_video.py --base_prompt "$BASE_PROMPT" --inpaint_prompt "$INPAINT_PROMPT" --logpath "$logpath" --mode "rotate"
python make_inpaint_seed_video.py --base_prompt "$BASE_PROMPT" --inpaint_prompt "$INPAINT_PROMPT" --logpath "$logpath" --mode "optss"
python make_inpaint_seed_video.py --base_prompt "$BASE_PROMPT" --inpaint_prompt "$INPAINT_PROMPT" --logpath "$logpath" --mode "optslat"