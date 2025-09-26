#!/bin/bash

device_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
# INCLUDE_STR="localhost:$CUDA_VISIBLE_DEVICES"
# INCLUDE_STR="localhost:0,1,2,3"
INCLUDE_STR="localhost:$(seq -s, 0 $((device_num-1)))"
# unset CUDA_VISIBLE_DEVICES

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --run_name babyllava_resnext_gpt2_SAYCam_phase1_dev \
    --model_name_or_path checkpoints_dev/gpt2_lr0.0005_bs32_epoch10 \
    --version plain \
    --data_path ./playground/data/SAYCam/llava_train_data_gt0.2.json \
    --image_folder ./playground/data/SAYCam/train_5fps \
    --vision_tower eminorhan/dino_sfp_resnext50 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints_dev \
    --num_train_epochs 5 \
    --per_device_train_batch_size 40 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 3e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb