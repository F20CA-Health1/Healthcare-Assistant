#!/bin/bash

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"

# change model name here
model="m42-health/Llama3-Med42-8B"

selected_subjects="all"
gpu_util=0.8

cd ../../

export CUDA_VISIBLE_DEVICES=0


python evaluate_from_mmlu_college_medicine.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util
