#!/bin/bash

# Navigate to the correct folder where the script is located
#cd ~/cacao-transformers/transformers/examples/pytorch/image-classification

# Set the dataset and output directory
dataset="SemilleroCV/Cocoa-dataset"
output_dir="./cocoa_outputs_resnet/"

# Run the model training script with ResNet50
python run_image_classification.py \
    --dataset_name $dataset \
    --output_dir $output_dir \
    --remove_unused_columns False \
    --label_column_name label \
    --ignore_mismatched_sizes True \
    --do_train \
    --do_eval \
    --model_name_or_path google/efficientnet-b0 \
    --push_to_hub \
    --push_to_hub_model_id efficientnet-b0-cocoa \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
