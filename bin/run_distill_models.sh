#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# This contains sections on
# (a) Fusion of 2 models into different architecture (MLPSmall and MLPLarge)
# (b) Distillation of MLPNet into MLPSmall
# Uncomment individual sections to run the corresponding experiments.

################## FUSION INTO DIFFERENT ARCHITECTURE ###################
# The current code does fusion into MLPLarge.
# For fusion into MLPSmall - change the experiment name from large to small
# and, change the hidden dims to 200, 100, 50.

base_model_prefix="mlp_sgd_models_layer_3"
layer=3

################ TLP Fusion ###################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="tlp"
#  tlp_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_distill_large_${fusion_type}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'FC' \
#                --input_dim 784 \
#                --hidden_dims 800 400 200 \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --use_pre_activations \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --tlp_init_type 'distill' \
#                --tlp_init_model 0 \
#                --tlp_ot_solver 'sinkhorn' \
#                --tlp_sinkhorn_regularization "${reg}" \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth"
#done

#fusion_type="tlp"
#tlp_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_distill_large_${fusion_type}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 128 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 800 400 200 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 256 \
#              --use_pre_activations \
#              --tlp_cost_choice "${tlp_cost_choice}" \
#              --tlp_init_type 'distill' \
#              --tlp_init_model 0 \
#              --tlp_ot_solver 'emd' \
#              --tlp_sinkhorn_regularization 0 \
#              --model_path_list \
#              "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#              "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth"


################# OT Fusion ####################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="ot"
#  ot_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_distill_large_${fusion_type}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'FC' \
#                --input_dim 784 \
#                --hidden_dims 800 400 100 \
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
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth"
#done

#fusion_type="ot"
#ot_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_distill_large_${fusion_type}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 128 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 800 400 200 \
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
#              --model_path_list \
#              "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#              "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth"



################ DISTILLATION INTO DIFFERENT ARCHITECTURE #################

# This code does one shot distillation of MLPNet into MLPSmall type model.
# Note down the results as each experiment for each hyperparameter finishes.

#prefix="${base_model_prefix}_one_shot_distill"
#for seed in 437 348 233 82 31
#do
#  fusion_type="tlp"
#  tlp_cost_choice="weight"
#  for reg in 0.01 0.005 0.001 0.0005
#  do
#    python src/tlp_model_fusion/fuse_models.py \
#                  --experiment_name "${prefix}_small_${fusion_type}_${seed}" \
#                  --dataset_name 'MNISTNorm' \
#                  --batch_size 128 \
#                  --model_name 'FC' \
#                  --input_dim 784 \
#                  --hidden_dims 200 100 50 \
#                  --output_dim 10 \
#                  --num_epochs 20 \
#                  --seed "43" \
#                  --gpu_ids "1" \
#                  --fusion_type "$fusion_type" \
#                  --activation_batch_size 256 \
#                  --use_pre_activations \
#                  --tlp_cost_choice "${tlp_cost_choice}" \
#                  --tlp_ot_solver 'sinkhorn' \
#                  --tlp_sinkhorn_regularization "${reg}" \
#                  --model_path_list \
#                  "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_${seed}/snapshots/best_val_acc_model.pth"
#  done
#  # EMD
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_small_${fusion_type}_${seed}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'FC' \
#                --input_dim 784 \
#                --hidden_dims 200 100 50 \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --use_pre_activations \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --tlp_ot_solver 'emd' \
#                --tlp_sinkhorn_regularization "${reg}" \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_${seed}/snapshots/best_val_acc_model.pth"
#
#  fusion_type="ot"
#  ad_hoc_cost_choice="weight"
#  for reg in 0.01 0.005 0.001 0.0005
#  do
#    python src/tlp_model_fusion/fuse_models.py \
#                  --experiment_name "${prefix}_small_${fusion_type}_${seed}" \
#                  --dataset_name 'MNISTNorm' \
#                  --batch_size 128 \
#                  --model_name 'FC' \
#                  --input_dim 784 \
#                  --hidden_dims 200 100 50 \
#                  --output_dim 10 \
#                  --num_epochs 20 \
#                  --seed "43" \
#                  --gpu_ids "1" \
#                  --fusion_type "$fusion_type" \
#                  --activation_batch_size 256 \
#                  --use_pre_activations \
#                  --ad_hoc_cost_choice "${ad_hoc_cost_choice}" \
#                  --ad_hoc_ot_solver 'sinkhorn' \
#                  --ad_hoc_sinkhorn_regularization "${reg}" \
#                  --model_path_list \
#                  "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_${seed}/snapshots/best_val_acc_model.pth"
#  done
#  # EMD
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_small_${fusion_type}_${seed}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'FC' \
#                --input_dim 784 \
#                --hidden_dims 200 100 50 \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --use_pre_activations \
#                --ad_hoc_cost_choice "${ad_hoc_cost_choice}" \
#                --ad_hoc_ot_solver 'emd' \
#                --ad_hoc_sinkhorn_regularization "${reg}" \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_${seed}/snapshots/best_val_acc_model.pth"
#done
