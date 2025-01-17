#!/bin/bash

#SBATCH --account=dl_jobs
#SBATCH --job-name=$JOB_NAME
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=$JOB_NAME.out          # Standard output and error log
#SBATCH --time=10:00:00            # Wall time limit (hh:mm:ss)

export PYTHONPATH=$PYTHONPATH:.

################## FUSION INTO SAME ARCHITECTURE ###################

# To change for experiments: dataset_name, hidden_dims
# You can also use the Sinkhorn solver for OT instead of EMD.
fusion_type="ot"
ot_cost_choice="weight"
python src/fuse_models.py \
              --experiment_name "${prefix}_model_fusion_${fusion_type}" \
              --dataset_name 'CIFAR10' \
              --batch_size 128 \
              --model_name 'FC' \
              --input_dim 3072 \
              --hidden_dims 1024 512 256 \
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
