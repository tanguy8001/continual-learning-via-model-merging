#!/bin/bash

#SBATCH --account=dl_jobs
#SBATCH --job-name=$JOB_NAME
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=$JOB_NAME.out          # Standard output and error log
#SBATCH --time=10:00:00            # Wall time limit (hh:mm:ss)

export PYTHONPATH=$PYTHONPATH:.

################## FUSION INTO SAME ARCHITECTURE ###################

# Uncomment individual sections to run the experiment of fusion.
# Comment out appropriate number of models to perform model fusion for 2, 4 and 6 models.

base_model_prefix="mlp_sgd_models_layer_3"
layer=3

################## TLP Fusion #####################

#for reg in 0.0005
#do
# fusion_type="tlp"
# tlp_cost_choice="weight"
# python src/tlp_model_fusion/fuse_models.py \
#               --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#               --dataset_name 'MNISTNorm' \
#               --batch_size 128 \
#               --model_name 'FC' \
#               --input_dim 784 \
#               --hidden_dims 800 400 200 \
#               --output_dim 10 \
#               --num_epochs 20 \
#               --seed "43" \
#               --gpu_ids "0" \
#               --fusion_type "$fusion_type" \
#               --activation_batch_size 256 \
#               --use_pre_activations \
#               --tlp_cost_choice "${tlp_cost_choice}" \
#               --tlp_init_type 'identity' \
#               --tlp_init_model 0 \
#               --tlp_ot_solver 'sinkhorn' \
#               --tlp_sinkhorn_regularization "${reg}" \
#               --model_path_list \
#               "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth" \
#               "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_43/snapshots/best_val_acc_model.pth" 
#done

# for reg in 0.01 0.005 0.001 0.0005
# do
#  fusion_type="tlp"
#  tlp_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
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
#                --use_pre_activations \
#                --tlp_cost_choice "${tlp_cost_choice}" \
#                --tlp_init_type 'identity' \
#                --tlp_init_model 0 \
#                --tlp_ot_solver 'sinkhorn' \
#                --tlp_sinkhorn_regularization "${reg}" \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_233/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_82/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_31/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_786/snapshots/best_val_acc_model.pth"
# done

#fusion_type="tlp"
#tlp_cost_choice="weight"
#python src/tlp_model_fusion/fuse_models.py \
#              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 128 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 400 200 100 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --seed "43" \
#              --gpu_ids "1" \
#              --fusion_type "$fusion_type" \
#              --activation_batch_size 128 \
#              --use_pre_activations \
#              --tlp_cost_choice "${tlp_cost_choice}" \
#              --tlp_init_type 'identity' \
#              --tlp_init_model 0 \
#              --tlp_ot_solver 'emd' \
#              --tlp_sinkhorn_regularization 0 \
#              --model_path_list \
#              "FC,/home/tdieudonne/dl2/model_A/final_model.pth" \
#              "FC,/home/tdieudonne/dl2/model_B/final_model.pth" \


################## OT Fusion #####################

#for reg in 0.01 0.005 0.001 0.0005
#do
#  fusion_type="ot"
#  ot_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
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
#                --ad_hoc_cost_choice "${ot_cost_choice}" \
#                --use_pre_activations \
#                --ad_hoc_ot_solver 'sinkhorn' \
#                --ad_hoc_sinkhorn_regularization "${reg}" \
#                --ad_hoc_initialization 0 \
#                --model_path_list \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_437/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_233/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_82/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_31/snapshots/best_val_acc_model.pth" \
#                "FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_786/snapshots/best_val_acc_model.pth"
#done

fusion_type="curve"
ot_cost_choice="weight"
python src/tlp_model_fusion/fuse_models.py \
              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
              --dataset_name 'MNISTNorm' \
              --batch_size 128 \
              --model_name 'FC' \
              --input_dim 784 \
              --hidden_dims 800 400 200 \
              --output_dim 10 \
              --num_epochs 20 \
              --seed "43" \
              --gpu_ids "1" \
              --fusion_type "$fusion_type" \
              --activation_batch_size 256 \
              --ad_hoc_cost_choice "${ot_cost_choice}" \
              --use_pre_activations \
              --ad_hoc_ot_solver 'emd' \
              --ad_hoc_sinkhorn_regularization 0 \
              --ad_hoc_initialization 0 \
              --model_path_list \
              "FC,/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_A/final_model.pth" \
              "FC,/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_B/final_model.pth" \
              #"FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_233/snapshots/best_val_acc_model.pth" \
              #"FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_82/snapshots/best_val_acc_model.pth" \
              #"FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_31/snapshots/best_val_acc_model.pth" \
              #"FC,result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_786/snapshots/best_val_acc_model.pth"


################## Vanilla Averaging Fusion #####################

#fusion_type="curve"
#python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
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
#                --model_path_list \
#                "FC,/home/tdieudonne/dl2/model_A/final_model.pth" \
#                "FC,/home/tdieudonne/dl2/model_B/final_model.pth" \
