#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

######## Generate Grid Plane for FC Model########

# base_model_prefix="mlp_sgd_models_layer_3"
# layer=3

python src/tlp_model_fusion/plane.py \
	--experiment_name 'visualization' \
	--dataset_name 'MNISTNorm' \
	--batch_size 128 \
	--model_name 'FC' \
	--model 'FCModel' \
	--input_dim 784 \
	--hidden_dims 800 400 200 \
	--output_dim 10 \
	--seed "43" \
	--gpu_ids '0' \
	--init_start "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_A/final_model.pth" \
	--init_end "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_B/final_model.pth" \
	--fused_model_path "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/final_curve_fusion_model.pth" \
	--curve_ckpt "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/final_curve_model.pth" \
	--num_bends 3 \
	--curve="Bezier" \
    --grid_points 21 \
    --margin_left 0.25 \
    --margin_right 0.25 \
    --margin_top 0.25 \
    --margin_bottom 0.25

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

