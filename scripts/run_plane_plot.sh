#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# Uncomment individual sections to run the experiments.

#### Visualization for FC model with MNIST #####
 python src/plane_plot.py \
 	--experiment_name "visualization" \
 	--model_name "FC" \
        --model "FCModel" \
 	--dataset_name "CIFAR10" \
 	--result_path "/home/tdieudonne/dl3/src/checkpoints" \
        --curve_ckpt "/home/tdieudonne/dl3/src/checkpoints/final_curve_model.pth" \
        --tr_vmax 0.4 \
        --tr_log_alpha -5.0 \
        --te_vmax 8.0 \
        --te_log_alpha -2.0 \

