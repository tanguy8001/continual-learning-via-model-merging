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






# ##### Visualization for VGG11 with CIFAR10 #####
# python src/tlp_model_fusion/plane_plot.py \
# 		--experiment_name "visualization" \
# 		--model_name "vgg11" \
# 		--dataset_name "CIFAR10" \
# 		--result_path "result" \
#     --tr_vmax 1.4 \
#     --tr_log_alpha -6.0 \
#     --te_vmax 20.0 \
#     --te_log_alpha -2.0

# ##### Visualization for VGG11 with CIFAR10 (finetuned) #####
# python src/tlp_model_fusion/plane_plot.py \
# 		--experiment_name "finetune_visualization" \
# 		--model_name "vgg11" \
# 		--dataset_name "CIFAR10" \
# 		--result_path "result" \
#     --tr_vmax 1.4 \
#     --tr_log_alpha -6.0 \
#     --te_vmax 20.0 \
#     --te_log_alpha -2.0

##### Visualization for ResNet with CIFAR10 #####
# python src/tlp_model_fusion/plane_plot.py \
# 		--experiment_name "visualization" \
# 		--model_name "resnet18" \
# 		--dataset_name "CIFAR10" \
# 		--result_path "result" \
#     --tr_vmax 1.5 \
#     --tr_log_alpha -6.0 \
#     --te_vmax 24.0 \
#     --te_log_alpha -2.0

##### Visualization for ResNet with CIFAR10 (finetuned) #####
# python src/tlp_model_fusion/plane_plot.py \
# 		--experiment_name "finetune_visualization" \
# 		--model_name "resnet18" \
# 		--dataset_name "CIFAR10" \
# 		--result_path "result" \
#     --tr_vmax 0.8 \
#     --tr_log_alpha -6.0 \
#     --te_vmax 24.0 \
#     --te_log_alpha -2.0


###### Visualization for RNN with MNIST #####
#python src/tlp_model_fusion/plane_plot.py \
#    --experiment_name "visualization" \
#    --model_name "RNN" \
#    --dataset_name "SplitMNIST" \
#    --result_path "result" \
#    --tr_vmax 0.01 \
#    --tr_log_alpha -8.0 \
#    --te_vmax 15.0 \
#    --te_log_alpha -2.0

##### Visualization for LSTM with MNIST #####
# python src/tlp_model_fusion/plane_plot.py \
#     --experiment_name "visualization" \
#     --model_name "LSTM" \
#     --dataset_name "SplitMNIST" \
#     --result_path "result" \
#     --tr_vmax 0.005 \
#     --tr_log_alpha -8.0 \
#     --te_vmax 10.0 \
#     --te_log_alpha -2.0

