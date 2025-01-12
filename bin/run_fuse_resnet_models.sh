#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.


base_model_prefix="resnet18_nmp_v1"
prefix="${base_model_prefix}_fusion"

# Uncomment individual section to run the experiments.

################# TLP Fusion ######################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="tlp"
#  tlp_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 256 \
#                --model_name 'resnet18' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --use_pre_activations \
#                --tlp_init_type 'identity' \
#                --tlp_init_model 0 \
#                --tlp_ot_solver 'sinkhorn' \
#                --tlp_sinkhorn_regularization "${reg}" \
#                --resnet_skip_connection_handling "pre" \
#                --model_path_list \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#done

for reg in 0.0005
do
 fusion_type="tlp"
 tlp_cost_choice="weight"
 python src/tlp_model_fusion/fuse_models.py \
               --experiment_name "${prefix}_model_fusion_${fusion_type}" \
               --dataset_name 'CIFAR10' \
               --batch_size 256 \
               --model_name 'resnet18' \
               --output_dim 10 \
               --num_epochs 20 \
               --seed "43" \
               --gpu_ids "0" \
               --fusion_type "$fusion_type" \
               --activation_batch_size 256 \
               --tlp_cost_choice "${tlp_cost_choice}" \
               --use_pre_activations \
               --tlp_init_type 'identity' \
               --tlp_init_model 0 \
               --tlp_ot_solver 'sinkhorn' \
               --tlp_sinkhorn_regularization "${reg}" \
               --resnet_skip_connection_handling "pre" \
               --model_path_list \
               "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
               "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
done


#fusion_type="tlp"
#tlp_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 256 \
#              --model_name 'resnet18' \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 256 \
#              --tlp_cost_choice "${tlp_cost_choice}" \
#              --use_pre_activations \
#              --tlp_init_type 'identity' \
#              --tlp_init_model 0 \
#              --tlp_ot_solver 'emd' \
#              --tlp_sinkhorn_regularization 0 \
#              --resnet_skip_connection_handling "pre" \
#              --model_path_list \
#              "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#              "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"



################# OT Fusion ######################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="ot"
#  ot_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 256 \
#                --model_name 'resnet18' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --ad_hoc_cost_choice "${ot_cost_choice}" \
#                --use_pre_activations \
#                --ad_hoc_ot_solver 'sinkhorn' \
#                --ad_hoc_sinkhorn_regularization "${reg}" \
#                --ad_hoc_initialization 0 \
#                --resnet_skip_connection_handling 'pre' \
#                --model_path_list \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#done


#fusion_type="ot"
#ot_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 256 \
#              --model_name 'resnet18' \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 256 \
#              --ad_hoc_cost_choice "${ot_cost_choice}" \
#              --use_pre_activations \
#              --ad_hoc_ot_solver 'emd' \
#              --ad_hoc_sinkhorn_regularization 0 \
#              --ad_hoc_initialization 0 \
#              --resnet_skip_connection_handling 'pre' \
#              --model_path_list \
#              "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#              "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"



################# Vanilla Averaging Fusion ######################

#fusion_type="avg"
#python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 256 \
#                --model_name 'resnet18' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --model_path_list \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "resnet18,result/${base_model_prefix}/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
