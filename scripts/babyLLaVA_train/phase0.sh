#!/bin/bash

accelerate launch llava/train/train_babylm.py \
    --arch gpt2 \
    --tokenizer bpe \
    --lr 0.0005 \
    --gacc 1 \
    --bs 8 \
    --epoch 10 \
    --output_dir ./checkpoints \