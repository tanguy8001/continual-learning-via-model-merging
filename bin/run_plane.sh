#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

######## Generate Grid Plane for FC Model########

# base_model_prefix="mlp_sgd_models_layer_3"
# layer=3

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'visualization' \
# 	      	--dataset_name 'MNISTNorm' \
# 		      --batch_size 128 \
# 		      --model_name 'FC' \
# 		      --input_dim 784 \
# 		      --hidden_dims 800 400 200 \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_348/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/${base_model_prefix}/FC_MNISTNorm/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/_model_fusion_tlp/FC_MNISTNorm/runs/fusion_tlp_num_models_2_layers_3_seed_43_cost_choice_weight_solver_sinkhorn_preact_init_identity_model_0_reg_0.0005/snapshots/fused_model.pth" \
#           --grid_points 21 \
#           --margin_left 0.25 \
#           --margin_right 0.25 \
#           --margin_top 0.25 \
#           --margin_bottom 0.25

######### Generate Grid Plane for Hetero MNIST ########

# base_model_prefix="mlp_hetero"
# layer=3

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'visualization' \
# 	      	--dataset_name 'MNISTNorm' \
#           --heterogeneous \
# 		      --batch_size 128 \
# 		      --model_name 'FC' \
# 		      --input_dim 784 \
# 		      --hidden_dims 400 200 100 \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_786special_dig_4_special_train/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/${base_model_prefix}/FC_HeteroMNIST/runs/debug_seed_786special_dig_4/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/mlp_hetero_hetero_distill_model_fusion_tlp_layer__786_0.9/FC_MNISTNorm/runs/fusion_tlp_num_models_2_layers_3_seed_43_cost_choice_weight_solver_sinkhorn_init_distill_model_0_reg_0.005/snapshots/fused_model.pth" \
#           --grid_points 41 \
#           --margin_left 0.4 \
#           --margin_right 0.4 \
#           --margin_top 0.2 \
#           --margin_bottom 0.2

######### Generate Grid Plane for VGG11 model #########

# base_model_prefix="deepcnn"

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'visualization' \
# 	      	--dataset_name 'CIFAR10' \
# 		      --batch_size 128 \
# 		      --model_name 'vgg11' \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/${base_model_prefix}/vgg11_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/vgg11_cnn_model_fusion_tlp/vgg11_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_init_identity_model_0_reg_0.0005/snapshots/fused_model.pth"\
#           --grid_points 21 \
#           --margin_left 0.25 \
#           --margin_right 0.25 \
#           --margin_top 0.25 \
#           --margin_bottom 0.25


######## Generate Grid Plane for VGG11 model (finetune) #########

# base_model_prefix="deepcnn"

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'finetune_visualization' \
# 	      	--dataset_name 'CIFAR10' \
# 		      --batch_size 128 \
# 		      --model_name 'vgg11' \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/vgg_finetune_model1/vgg11_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/vgg_finetune_model2/vgg11_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/vgg11_cnn_model_fusion_tlp/vgg11_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_init_identity_model_0_reg_0.0005/snapshots/fused_model.pth"\
#           --finetune_visualization \
#           --finetuned_model_path "result/vgg_finetune_tlp_model/vgg11_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
#           --grid_points 21 \
#           --margin_left 0.25 \
#           --margin_right 0.25 \
#           --margin_top 0.25 \
#           --margin_bottom 0.25



######### Generate Grid Plane for ResNet18 model #########

# base_model_prefix="resnet18_nmp_v1"

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'visualization' \
# 	      	--dataset_name 'CIFAR10' \
# 		      --batch_size 128 \
# 		      --model_name 'resnet18' \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/resnet18_nmp_v1/resnet18_CIFAR10/runs/debug_seed_786/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/resnet18_nmp_v1/resnet18_CIFAR10/runs/debug_seed_234/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/resnet18_nmp_v1_fusion_model_fusion_tlp/resnet18_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_preact_init_identity_model_0_reg_0.0005_skip_conn_pre/snapshots/fused_model.pth"\
#           --grid_points 21 \
#           --margin_left 0.15 \
#           --margin_right 0.15 \
#           --margin_top 0.15 \
#           --margin_bottom 0.15


####### Generate Grid Plane for ResNet model (finetune) #########

# base_model_prefix="resnet18_nmp_v1"

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'finetune_visualization' \
# 	      	--dataset_name 'CIFAR10' \
# 		      --batch_size 128 \
# 		      --model_name 'resnet18' \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/resnet18_nmp_v1_finetune_786/resnet18_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
# 		      --init_end "result/resnet18_nmp_v1_finetune_234/resnet18_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
# 		      --fused_model_path "result/resnet18_nmp_v1_fusion_model_fusion_tlp/resnet18_CIFAR10/runs/fusion_tlp_num_models_2_layers_0_seed_43_cost_choice_weight_solver_sinkhorn_preact_init_identity_model_0_reg_0.0005_skip_conn_pre/snapshots/fused_model.pth"\
#           --finetune_visualization \
#           --finetuned_model_path "result/resnet18_nmp_v1_finetune_tlp/resnet18_CIFAR10/runs/debug_seed_43/snapshots/best_val_acc_model.pth" \
#           --grid_points 21 \
#           --margin_left 0.15 \
#           --margin_right 0.15 \
#           --margin_top 0.15 \
#           --margin_bottom 0.15


######### Generate Grid Plane for RNN model #############

# base_model_prefix="image_rnn_128"
# layer=3

# python src/tlp_model_fusion/plane.py \
#           --experiment_name 'visualization' \
# 	      	--dataset_name 'SplitMNIST' \
# 		      --batch_size 64 \
# 		      --model_name 'RNN' \
# 		      --input_dim 28 \
# 		      --hidden_dims 128 \
# 		      --output_dim 10 \
# 		      --seed "43" \
# 		      --gpu_ids '0' \
# 		      --init_start "result/layer_1_dim_256_scale_1_adam/rnn_mnist/dataset_1/best_val_acc_model.pth" \
# 		      --init_end "result/layer_1_dim_256_scale_1_adam/rnn_mnist/dataset_2/best_val_acc_model.pth" \
# 		      --fused_model_path "result/test2_adam/rnn_256_mnist_split/layer_1/idenitity/models_1_2_reg_0.001.pth" \
#           --permuted_model_path "result/test2_adam/rnn_256_mnist_split/layer_1/idenitity/permuted_model_2.pth" \
#           --grid_points 21 \
#           --margin_left 0.2 \
#           --margin_right 0.2 \
#           --margin_top 0.2 \
#           --margin_bottom 0.2

######### Generate Grid Plane for LSTM model #############

#base_model_prefix="image_lstm_128"
#layer=3
#
#python src/tlp_model_fusion/plane.py \
#          --experiment_name 'visualization' \
#	      	--dataset_name 'SplitMNIST' \
#		      --batch_size 64 \
#		      --model_name 'LSTM' \
#		      --input_dim 28 \
#		      --hidden_dims 128 \
#		      --output_dim 10 \
#		      --seed "43" \
#		      --gpu_ids '0' \
#		      --init_start "result/layer_1_dim_256_scale_1_adam/lstm_mnist/dataset_1/best_val_acc_model.pth" \
#		      --init_end "result/layer_1_dim_256_scale_1_adam/lstm_mnist/dataset_2/best_val_acc_model.pth" \
#		      --fused_model_path "result/test2_adam/lstm_256_mnist_split/layer_1/idenitity/models_1_2_reg_0.0032.pth" \
#          --permuted_model_path "result/test2_adam/lstm_256_mnist_split/layer_1/idenitity/permuted_model_2.pth" \
#          --grid_points 31 \
#          --margin_left 0.2 \
#          --margin_right 0.2 \
#          --margin_top 0.2 \
#          --margin_bottom 0.2 




