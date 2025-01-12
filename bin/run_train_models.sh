#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src/tlp_model_fusion'


# Uncomment individual section to run the experiments.

############ FC Fusion models training #############

#MLPNet Training below

# layer=3
# for seed in 43 348
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_sgd_models_layer_${layer}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 800 400 200 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.05 \
#              --seed "$seed" \
#              --gpu_ids "0"
# done


# layer=3
# for seed in 43 348 437 82 233 31 786 234 12 7
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_sgd_models_layer_${layer}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 400 200 100 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.05 \
#              --seed "$seed" \
#              --gpu_ids "1"
# done

# MLPSmall Training below

#layer=3
#for seed in 43 348 437 82 233 31 786 234 12 7
#do
#  python "${SRC}"/train_models.py \
#              --experiment_name "small_mlp_sgd_models_layer_${layer}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 200 100 50 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.05 \
#              --seed "$seed" \
#              --gpu_ids "1"
#done

# MLPLarge Training below

#layer=3
#for seed in 43 348 437 82 233 31 786 234 12 7
#do
#  python "${SRC}"/train_models.py \
#              --experiment_name "large_mlp_sgd_models_layer_${layer}" \
#              --dataset_name 'MNISTNorm' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 800 400 200 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.05 \
#              --seed "$seed" \
#              --gpu_ids "1"
#done


############ VGG11 Fusion models training #############

# for seed in 786, 234
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "deepcnn" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 128 \
#              --model_name 'vgg11' \
#              --output_dim 10 \
#              --num_epochs 300 \
#              --optimizer 'SGD' \
#              --lr 0.05 \
#              --momentum 0.9 \
#              --lr_scheduler 'StepLR' \
#              --lr_gamma 0.5 \
#              --lr_step_size 30 \
#              --weight_decay 5e-4 \
#              --seed "$seed" \
#              --gpu_ids "0"
# done


############ ResNet18 Fusion models training #############

# for seed in 786 234
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "resnet18_nmp_v1" \
#              --dataset_name 'CIFAR10' \
#              --batch_size 128 \
#              --model_name 'resnet18' \
#              --output_dim 10 \
#              --num_epochs 300 \
#              --optimizer 'SGD' \
#              --lr 0.1 \
#              --momentum 0.9 \
#              --lr_scheduler 'MultiStepLR' \
#              --lr_milestones 150 \
#              --lr_gamma 0.1 \
#              --weight_decay 0.0001 \
#              --seed "$seed" \
#              --gpu_ids "0"
# done


############ FC models training for heterogeneous #############

## Training with special digit

# for seed in 786 234 7 12 31
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_hetero" \
#              --dataset_name 'HeteroMNIST' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 200 100 50 \
#              --output_dim 10 \
#              --num_epochs 10 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.01 \
#              --seed "$seed" \
#              --hetero_special_digit 4 \
#              --hetero_special_train \
#              --hetero_other_digits_train_split 0.9 \
#              --gpu_ids "1"
# done



## Training with special digit

# for seed in 786
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_hetero" \
#              --dataset_name 'HeteroMNIST' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 400 200 100 \
#              --output_dim 10 \
#              --num_epochs 10 \
#              --optimizer 'SGD' \
#              --momentum 0.5 \
#              --lr 0.01 \
#              --seed "$seed" \
#              --hetero_special_digit 4 \
#              --hetero_special_train \
#              --hetero_other_digits_train_split 0.9 \
#              --gpu_ids "0"
# done


## Training without special digit

#for seed in 786 234 7 12 31
#do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_hetero" \
#              --dataset_name 'HeteroMNIST' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 200 100 50 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.9 \
#              --lr 0.01 \
#              --seed "$seed" \
#              --hetero_special_digit 4 \
#              --hetero_other_digits_train_split 0.9 \
#              --gpu_ids "1"
#done

## Training without special digit

# for seed in 786
# do
#  python "${SRC}"/train_models.py \
#              --experiment_name "mlp_hetero" \
#              --dataset_name 'HeteroMNIST' \
#              --batch_size 64 \
#              --model_name 'FC' \
#              --input_dim 784 \
#              --hidden_dims 400 200 100 \
#              --output_dim 10 \
#              --num_epochs 20 \
#              --optimizer 'SGD' \
#              --momentum 0.9 \
#              --lr 0.01 \
#              --seed "$seed" \
#              --hetero_special_digit 4 \
#              --hetero_other_digits_train_split 0.9 \
#              --gpu_ids "0"
# done

########## Run RNN model training #########

# SRC='src/tlp_rnn_fusion'

# seeds=(847 53)
# for idx in 1 2
# do
#   path="./result/layer_1_dim_256_scale_1_adam/rnn_mnist/dataset_${idx}"
#   mkdir -p "${path}"
#   python ${SRC}/train_rnn_mnist.py \
#         --model_name 'rnn' \
#         --dataset_name 'SplitMNIST' \
#         --nsplits 1 \
#         --split_index 1 \
#         --ds_scale_factor 1.0 \
#         --seed ${seeds[$idx-1]} \
#         --num_epochs 20 \
#         --learning_rate 0.001 \
#         --batch_size 64 \
#         --device 'cuda' \
#         --model_load_path '' \
#         --model_save_path "${path}" \
#         --vocab_size 10 \
#         --embed_dim 28 \
#         --hidden_dims [256] \
#         --momentum 0.9 \
#         --optimizer 'Adam' \
#         --weight_decay 0.00001
# done

########## Run LSTM model training #########

SRC='src/tlp_rnn_fusion'
for seed in 847 53 43 348 437 82 233 31 786 234
do
  path="./result/layer_1_dim_256_scale_1_adam_20/lstm_mnist/seed_${seed}"
  mkdir -p "${path}"
  python ${SRC}/train_rnn_mnist.py \
        --model_name 'lstm' \
        --dataset_name 'SplitMNIST' \
        --nsplits 1 \
        --split_index 1 \
        --ds_scale_factor 1.0 \
        --seed ${seed} \
        --num_epochs 20 \
        --learning_rate 0.001 \
        --batch_size 64 \
        --device 'cuda' \
        --model_load_path '' \
        --model_save_path "${path}" \
        --vocab_size 10 \
        --embed_dim 28 \
        --hidden_dims [256] \
        --momentum 0.9 \
        --optimizer 'Adam' \
        --weight_decay 0.00001 
done












