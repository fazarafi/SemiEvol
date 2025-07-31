#/bin/bash

CUDA_VISIBLE_DEVICES=$1 python pipeline.py --task factuality --model llama3.2 --num_samples $3 --labeled_dataset_name $2
