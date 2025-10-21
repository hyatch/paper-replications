# In a git bash terminal, run ./launch_interface.sh to start the program

#!/bin/bash
set -e

# Configuration variables
# GO BACK to using the simple Hub ID!

# This accesses the HuggingFace Model
MODEL_PATH="google/paligemma-3b-pt-224"
# Prompt of the model to start with
PROMPT="The people are "
# Path to the image
IMAGE_FILE_PATH="citpsweikles.jpg"
# Maximum number of tokens to made by the model
MAX_TOKENS_TO_GENERATE=100
# A measure of how crazy the model should be
TEMPERATURE=0.8
# Which of the words should the model even consider
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

# run in the bash terminal
echo "Running inference..."
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE" \
    --only_cpu "$ONLY_CPU"
