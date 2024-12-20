#!/bin/bash

# Set the dataset and output directory
dataset="SemilleroCV/Cocoa-dataset"
output_dir="./cocoa_outputs_vgg/"

mkdir -p "$output_dir"

# Configure Weights & Biases (W&B)
export WANDB_PROJECT="cocoa-image-classification"
export WANDB_ENTITY="cristianrey"
export WANDB_RUN_NAME="vgg16-training-run"

# Run the training script with VGG16
python main.py \
    --dataset_name "$dataset" \
    --output_dir "$output_dir" \
    --with_tracking \
    --report_to wandb \
    --remove_unused_columns "false" \
    --label_column_name label \
    --ignore_mismatched_sizes \
    --do_train \
    --do_eval \
    --model_name_or_path Jaivin13/vgg16-face-expression-model \
    --push_to_hub \
    --push_to_hub_model_id "vgg16-cocoa" \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end "true" \
    --save_total_limit 3 \
    --seed 1337 \
    --run_name "vgg16-cocoa-run"
