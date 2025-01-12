#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

prefix="vgg11_cnn"
base_model_prefix="deepcnn"

# Uncomment individual sections to run the experiments.

############## TLP Fusion ###############

for reg in 0.0005
do
 fusion_type="tlp"
 tlp_cost_choice="weight"
 python src/tlp_model_fusion/fuse_models.py \
               --experiment_name "${prefix}_model_fusion_${fusion_type}" \
               --dataset_name 'CIFAR10' \
               --batch_size 128 \
               --model_name 'vgg11' \
               --output_dim 10 \
               --num_epochs 20 \
               --seed "43" \
               --gpu_ids "0" \
               --fusion_type "$fusion_type" \
               --activation_batch_size 256 \
               --tlp_cost_choice "${tlp_cost_choice}" \
               --tlp_init_type 'identity' \
               --tlp_init_model 0 \
               --tlp_ot_solver 'sinkhorn' \
               --tlp_sinkhorn_regularization "${reg}" \
               --model_path_list \
               "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
               "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
done

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="tlp"
#  tlp_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 128 \
#                --model_name 'vgg11' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --tlp_init_type 'identity' \
#                --tlp_init_model 0 \
#                --tlp_ot_solver 'sinkhorn' \
#                --tlp_sinkhorn_regularization "${reg}" \
#                --model_path_list \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#done

#fusion_type="tlp"
#tlp_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 128 \
#              --model_name 'vgg11' \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 256 \
#              --tlp_cost_choice "${tlp_cost_choice}" \
#              --tlp_init_type 'identity' \
#              --tlp_init_model 0 \
#              --tlp_ot_solver 'emd' \
#              --tlp_sinkhorn_regularization 0 \
#              --model_path_list \
#              "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#              "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"


############## OT Fusion ##################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="ot"
#  ot_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 128 \
#                --model_name 'vgg11' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --ad_hoc_cost_choice "${ot_cost_choice}" \
#                --ad_hoc_ot_solver 'sinkhorn' \
#                --ad_hoc_sinkhorn_regularization "${reg}" \
#                --ad_hoc_initialization 0 \
#                --model_path_list \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#done

#fusion_type="ot"
#ot_cost_choice="activation"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 128 \
#              --model_name 'vgg11' \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 256 \
#              --use_pre_activations \
#              --ad_hoc_cost_choice "${ot_cost_choice}" \
#              --ad_hoc_ot_solver 'emd' \
#              --ad_hoc_sinkhorn_regularization 0 \
#              --ad_hoc_initialization 0 \
#              --model_path_list \
#              "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#              "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"



################## VANILLA AVERAGING #################

#fusion_type="avg"
#python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'CIFAR10' \
#                --batch_size 128 \
#                --model_name 'vgg11' \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --model_path_list \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "vgg11,result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
