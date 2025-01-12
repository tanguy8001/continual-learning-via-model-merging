#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

################## FUSION INTO SAME ARCHITECTURE (RNN) ###################
# SRC='src/tlp_rnn_fusion'
# ds=(1 2)

# for idx in 0
# do
#   ds1=${ds[$idx]}
#   ds2=${ds[$idx+1]}
#   opt='adam'
#   #for reg in 0.05 0.04 0.02 0.01 0.005 0.002 0.001 0.0005
#   #for reg in 0.06 0.05 0.04 0.02 0.01
#   for reg in 0.001
#   do
#     path="result/test2_${opt}/rnn_256_mnist_split/layer_1/idenitity/"
#     mkdir -p ${path}
    
#     python ${SRC}/fuse_rnn_models.py \
#           --experiment_name test_fuse_01 \
#           --model_name RNN \
#           --dataset_name "SplitMNIST" \
#           --nsplits 1 \
#           --split_index 1 \
#           --ds_scale_factor 1.0 \
#           --input_dim 28 \
#           --embed_dim 28 \
#           --hidden_dims [256] \
#           --output_dim 10 \
#           --batch_size 64 \
#           --num_epochs 20 \
#           --seed 43 \
#           --gpu_ids 0 \
#           --optimizer 'Adam' \
#           --fusion_type tlp \
#           --alpha_h 1000.0 1.0 \
#           --theta_pi 1.0 \
#           --niters_rnn 200 \
#           --num_pi_iters 50 \
#           --activation_batch_size 256 \
#           --tlp_init_type 'identity' \
#           --tlp_cost_choice 'weight' \
#           --tlp_init_model 0 \
#           --tlp_ot_solver sinkhorn \
#           --tlp_sinkhorn_regularization $reg \
#           --tlp_sinkhorn_regularization_list ${reg} ${reg} ${reg} \
#           --save_path "${path}" \
#           --result_name "models_${ds1}_${ds2}_reg_${reg}.pth" \
#           --model_path_list \
#           RNN,"./result/layer_1_dim_256_scale_1_adam/rnn_mnist/dataset_${ds1}/best_val_acc_model.pth" \
#           RNN,"./result/layer_1_dim_256_scale_1_adam/rnn_mnist/dataset_${ds2}/best_val_acc_model.pth"

#   done
# done

# prefix="test"
# base_model_prefix="image_rnn_512"

# for reg in 0.005 0.001 0.0005
# do
#   fusion_type="gw"
#   tlp_cost_choice="weight"
#   python src/tlp_model_fusion/fuse_models.py \
#                 --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                 --dataset_name 'MNISTNorm' \
#                 --batch_size 128 \
#                 --model_name 'ImageRNN' \
#                 --input_dim 28 \
#                 --hidden_dims 512 \
#                 --output_dim 10 \
#                 --rnn_steps 28 \
#                 --num_epochs 20 \
#                 --seed "43" \
#                 --gpu_ids "0" \
#                 --fusion_type "$fusion_type" \
#                 --tlp_cost_choice "${tlp_cost_choice}" \
#                 --tlp_init_type 'identity' \
#                 --tlp_init_model 0 \
#                 --tlp_ot_solver 'sinkhorn' \
#                 --tlp_sinkhorn_regularization "${reg}" \
#                 --theta_w 1.0 \
#                 --theta_pi 1.0 \
#                 --auto_optimize 0 \
#                 --model_path_list \
#                 "ImageRNN,result/${base_model_prefix}/ImageRNN_MNISTNorm/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                 "ImageRNN,result/${base_model_prefix}/ImageRNN_MNISTNorm/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
# done


################## FUSION INTO SAME ARCHITECTURE (LSTM) ###################
SRC='src/tlp_rnn_fusion'
seed=(847 53 43 348 437 82 233 31 786 234)

for reg in 0.003
do
  for idx in {0..8}
  do
    seed1=${seed[$idx]}
    seed2=${seed[$idx+1]}
    opt='adam'
    path="result/tlp_${opt}/lstm_256_mnist_split_20/layer_1/idenitity/seed_${seed1}_${seed2}/"
    mkdir -p ${path}
    
    python ${SRC}/fuse_rnn_models.py \
          --experiment_name test_fuse_01 \
          --model_name 'LSTM' \
          --dataset_name "SplitMNIST" \
          --nsplits 1 \
          --split_index 1 \
          --ds_scale_factor 1.0 \
          --input_dim 28 \
          --embed_dim 28 \
          --hidden_dims [256] \
          --output_dim 10 \
          --batch_size 64 \
          --num_epochs 20 \
          --seed 43 \
          --gpu_ids 0 \
          --optimizer 'Adam' \
          --fusion_type tlp \
          --alpha_h 1000.0 1.0 \
          --theta_pi 1.0 \
          --num_pi_iters 50 \
          --niters_rnn 200 \
          --activation_batch_size 256 \
          --tlp_cost_choice 'weight' \
          --tlp_init_type 'identity' \
          --tlp_init_model 0 \
          --tlp_ot_solver sinkhorn \
          --tlp_sinkhorn_regularization $reg \
          --tlp_sinkhorn_regularization_list ${reg} ${reg} ${reg} \
          --save_path "${path}" \
          --result_name "models_reg_${reg}.pth" \
          --model_path_list \
          LSTM,"./result/layer_1_dim_256_scale_1_adam_20/lstm_mnist/seed_${seed1}/best_val_acc_model.pth" \
          LSTM,"./result/layer_1_dim_256_scale_1_adam_20/lstm_mnist/seed_${seed2}/best_val_acc_model.pth"

  done
done


#for reg in 0.005 0.001 0.0005
#do
#  fusion_type="ot"
#  ot_cost_choice="weight"
#  python src/tlp_model_fusion/fuse_models.py \
#                --experiment_name "${prefix}_model_fusion_${fusion_type}" \
#                --dataset_name 'MNISTNorm' \
#                --batch_size 128 \
#                --model_name 'ImageRNN' \
#                --input_dim 28 \
#                --hidden_dims 128 \
#                --output_dim 10 \
#                --rnn_steps 28 \
#                --num_epochs 20 \
#                --seed "43" \
#                --gpu_ids "0" \
#                --fusion_type "$fusion_type" \
#                --ad_hoc_cost_choice "${ot_cost_choice}" \
#                --use_pre_activations \
#                --ad_hoc_ot_solver 'sinkhorn' \
#                --ad_hoc_sinkhorn_regularization "${reg}" \
#                --ad_hoc_initialization 0 \
#                --model_path_list \
#                "ImageRNN,result/${base_model_prefix}/ImageRNN_MNISTNorm/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
#                "ImageRNN,result/${base_model_prefix}/ImageRNN_MNISTNorm/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#done