#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat
MODEL_VERSION=llava-med-v1.5-mistral-7b

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

python llava/train/test_1.py \
    --image-folder /home/user01/aiotlab/thaind/DAC001_CTAC3.75mm_H_1001_PETWB3DAC001 \
    --model-path /home/user01/aiotlab/thaind/LLaVA/checkpoints/llavamed-llava-med-v1.5-mistral-7b-pretrain-20epochs \
    --model-base ./checkpoints/$MODEL_VERSION \
    --question-file /home/user01/aiotlab/thaind/llavamed_test_data.json \
    --answers-file /home/user01/aiotlab/thaind/llavamed_test_data_infer.jsonl \