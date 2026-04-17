#/bin/bash!

BASE_PROMPT=$1 #"Trees on the grass"
OPTINIT_PROMPT=$2 #"A red roof house"
echo "Base prompt: $BASE_PROMPT"
echo "Optinit prompt: $OPTINIT_PROMPT"
logpath="logs/test_draft1_bottom"

python make_optinit_seed_video.py --base_prompt "$BASE_PROMPT" --optinit_prompt "$OPTINIT_PROMPT" --logpath "$logpath" --mode "rotate"
python make_optinit_seed_video.py --base_prompt "$BASE_PROMPT" --optinit_prompt "$OPTINIT_PROMPT" --logpath "$logpath" --mode "optss"
python make_optinit_seed_video.py --base_prompt "$BASE_PROMPT" --optinit_prompt "$OPTINIT_PROMPT" --logpath "$logpath" --mode "optslat"
