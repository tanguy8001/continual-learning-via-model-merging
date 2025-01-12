#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src/tlp_model_fusion'


################ FINETUNING EXPERIMENTS for DEEP CNN Models ################

# Uncomment the checkpoints and change the checkpoint_path argument to finetune specific model


################ VGG Finetuning #################

# checkpoint_path_tlp="result/vgg11_cnn_model_fusion_tlp/vgg11_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_init_identity_model_0_reg_0.0005/snapshots/fused_model.pth"
# checkpoint_path_ot="result/vgg11_cnn_model_fusion_ot/vgg11_CIFAR10/runs/fusion_ot_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_init_0_reg_0.001/snapshots/fused_model.pth"
# checkpoint_path_avg="result/vgg11_cnn_model_fusion_avg/vgg11_CIFAR10/runs/fusion_avg_num_models_2_layers_0_seed_43/snapshots/fused_model.pth"
# checkpoint_path_234="result/deepcnn/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
# checkpoint_path_786="result/deepcnn/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth"

# python "${SRC}"/train_models.py \
#            --experiment_name "vgg_finetune_model2" \
#            --dataset_name 'CIFAR10' \
#            --batch_size 128 \
#            --model_name 'vgg11' \
#            --output_dim 10 \
#            --num_epochs 100 \
#            --optimizer 'SGD' \
#            --lr 0.01 \
#            --momentum 0.9 \
#            --lr_scheduler 'StepLR' \
#            --lr_gamma 0.5 \
#            --lr_step_size 30 \
#            --weight_decay 5e-4 \
#            --seed "43" \
#            --gpu_ids "0" \
#            --load_checkpoint \
#            --checkpoint_path "$checkpoint_path_234"



################# Resnet Finetuning ###################

checkpoint_path_234="result/resnet18_nmp_v1/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth"
#checkpoint_path_786="result/resnet18_nmp_v1/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth"
#checkpoint_path_tlp="result/resnet18_nmp_v1_fusion_model_fusion_tlp/resnet18_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_preact_init_identity_model_0_reg_0.0005_skip_conn_pre/snapshots/fused_model.pth"
#checkpoint_path_ot="result/resnet18_nmp_v1_fusion_model_fusion_ot/resnet18_CIFAR10/runs/fusion_ot_num_models_2_layers_0_seed_43_cost_choice_weight_solver_emd_preact_init_0_skip_conn_pre/snapshots/fused_model.pth"
#checkpoint_path_avg="result/resnet18_nmp_v1_fusion_model_fusion_avg/resnet18_CIFAR10/runs/fusion_avg_num_models_2_layers_0_seed_43_skip_conn_pre/snapshots/fused_model.pth"

python "${SRC}"/train_models.py \
           --experiment_name "resnet18_nmp_v1_finetune_234" \
           --dataset_name 'CIFAR10' \
           --batch_size 128 \
           --model_name 'resnet18' \
           --output_dim 10 \
           --num_epochs 120 \
           --optimizer 'SGD' \
           --lr 0.1 \
           --momentum 0.9 \
           --lr_scheduler 'StepLR' \
           --lr_step_size 20 \
           --lr_gamma 0.5 \
           --weight_decay 0.0001 \
           --seed "43" \
           --gpu_ids "0" \
           --load_checkpoint \
           --checkpoint_path "$checkpoint_path_234"
