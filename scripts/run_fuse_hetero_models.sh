#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
base_model_prefix="mlp_hetero"
prefix="${base_model_prefix}_hetero_distill"

############ FUSION for Heterogeneous dataset #############

# Uncomment the fusion experiments for tlp to obtain results.
# Change WA and WB to obtain results for various proportions.
# Keep note of results for each seed while the fusion happens.

WA = 0.1
WB = 0.9
fusion_type="curve"
tlp_cost_choice="weight"
python src/tlp_model_fusion/fuse_models.py \
                --experiment_name "${prefix}_model_fusion_${fusion_type}_layer_${layer}_${seed}_${WA}" \
                --dataset_name 'MNISTNorm' \
                --batch_size 128 \
                --model_name 'FC' \
                --input_dim 784 \
                --hidden_dims 400 200 100 \
                --output_dim 10 \
                --num_epochs 20 \
                --seed "43" \
                --gpu_ids "1" \
                --fusion_type "$fusion_type" \
                --activation_batch_size 256 \
                --tlp_cost_choice "${tlp_cost_choice}" \
                --tlp_init_type 'distill' \
                --tlp_init_model 0 \
                --tlp_ot_solver 'emd' \
                --tlp_sinkhorn_regularization 0 \
                --model_weights "$WA" "$WB" \
                --model_path_list \
                "FC,C:\Users\tangu\Downloads\dnn-mode-connectivity-master\dnn-mode-connectivity-master\model_A\final_model.pth" \
                "FC,C:\Users\tangu\Downloads\dnn-mode-connectivity-master\dnn-mode-connectivity-master\model_B\final_model.pth" \

done

#for seed in 786
#do
# # The proportion of model A vs model B. Change to values such that WA + WB = 1.
# WA=0.9
# WB=0.1
#
# fusion_type="tlp"
# tlp_cost_choice="weight"
# for reg in 0.005
# do
#   python src/tlp_model_fusion/fuse_models.py \
#                 --experiment_name "${prefix}_model_fusion_${fusion_type}_layer_${layer}_${seed}_${WA}" \
#                 --dataset_name 'MNISTNorm' \
#                 --batch_size 128 \
#                 --model_name 'FC' \
#                 --input_dim 784 \
#                 --hidden_dims 400 200 100 \
#                 --output_dim 10 \
#                 --num_epochs 20 \
#                 --seed "43" \
#                 --gpu_ids "0" \
#                 --fusion_type "$fusion_type" \
#                 --activation_batch_size 256 \
#                 --tlp_cost_choice "${tlp_cost_choice}" \
#                 --tlp_init_type 'identity' \
#                 --tlp_init_model 0 \
#                 --tlp_ot_solver 'sinkhorn' \
#                 --tlp_sinkhorn_regularization "${reg}" \
#                 --model_weights "$WA" "$WB" \
#                 --model_path_list \
#                 "FC,result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_${seed}special_dig_4_special_train/snapshots/best_val_acc_model.pth" \
#                 "FC,result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_${seed}special_dig_4/snapshots/best_val_acc_model.pth"
# done
#done

#
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}_layer_${layer}_${seed}_${WA}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'FC' \
#                --input_dim 784 \
#                --hidden_dims 400 200 100 \
#                --output_dim 10 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "1" \
#                --fusion_type "$fusion_type" \
#                --activation_batch_size 256 \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --tlp_init_type 'distill' \
#                --tlp_init_model 0 \
#                --tlp_ot_solver 'emd' \
#                --tlp_sinkhorn_regularization 0 \
#                --model_weights "$WA" "$WB" \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_${seed}special_dig_4_special_train/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_${seed}special_dig_4/snapshots/best_val_acc_model.pth"
#
#done