#!/bin/bash

# Set the dataset and output directory
dataset="SemilleroCV/Cocoa-dataset"
output_dir="./cocoa_outputs_vit/"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Set W&B environment variables
export WANDB_PROJECT="cocoa-image-classification" 
export WANDB_ENTITY="cristianrey"   
export WANDB_RUN_NAME="vit-training-run"

# Run the model training script with VGG16 and W&B integration
python run_image_classification_no_trainer.py \
    --dataset_name "$dataset" \
    --output_dir "$output_dir" \
    --with_tracking \
    --report_to wandb \
    --remove_unused_columns "false" \
    --label_column_name label \
    --ignore_mismatched_sizes \
    --do_train \
    --do_eval \
    --model_name_or_path google/vit-base-patch16-224  \
    --push_to_hub \
    --push_to_hub_model_id vit-base-cocoa \
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
    --run_name "vit-cocoa-run"
