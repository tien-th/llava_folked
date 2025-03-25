#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION=llava-med-v1.5-mistral-7b
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# PROMPT_VERSION=plain
################## LLaMA-2 ##################

# --data_path /home/user01/aiotlab/thaind/data_desc_conv_train.json \
# --eval_data_path /home/user01/aiotlab/thaind/data_desc_conv_test.json \
    # --pretrain_mm_mlp_adapter /home/user01/aiotlab/thaind/LLaVA/checkpoints/ctvit_llavamed-llava-med-v1.5-mistral-7b-pretrain-1epochs/mm_projector.bin \
deepspeed llava/train/test_99.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /home/user01/aiotlab/thaind/data_desc_conv_train.json \
    --eval_data_path /home/user01/aiotlab/thaind/data_desc_conv_eval.json \
    --image_folder /home/user01/aiotlab/thaind/DAC001_CTAC3.75mm_H_1001_PETWB3DAC001 \
    --vision_tower /home/user10/huutien/simplified_anatomask/results/anatomask_100_epochs_8_batch_aug/encoder_only_9.pt \
    --pretrain_mm_mlp_adapter /home/user01/aiotlab/thaind/LLaVA/checkpoints/ctvit_projector_llavamed-llava-med-v1.5-mistral-7b-pretrain_RotateAug/checkpoint-1912/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/ctvit_llavamed-$MODEL_VERSION-finetune_RotateAug_lora_10_epochs_1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 2 \
    --report_to wandb
