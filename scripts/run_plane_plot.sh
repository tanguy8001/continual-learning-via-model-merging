#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# Uncomment individual sections to run the experiments.

#### Visualization for FC model with MNIST #####
 python src/tlp_model_fusion/plane_plot.py \
 		--experiment_name "visualization" \
 		--model_name "FC" \
    --model "FCModel" \
 		--dataset_name "MNISTNorm" \
 		--result_path "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints" \
    --curve_ckpt "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/final_curve_model.pth" \
     --tr_vmax 0.4 \
     --tr_log_alpha -5.0 \
     --te_vmax 8.0 \
     --te_log_alpha -2.0 \

##### Visualization for FC model with HeteroMNIST #####
# python src/tlp_model_fusion/plane_plot.py \
# 		--experiment_name "visualization" \
# 		--model_name "FC" \
# 		--dataset_name "HeteroMNIST" \
# 		--result_path "result" \
#     --tr_vmax 2.0 \
#     --tr_log_alpha -5.0 \
#     --te_vmax 20.0 \
#     --te_log_alpha -2.0 
