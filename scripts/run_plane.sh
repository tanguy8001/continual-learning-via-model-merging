#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

######## Generate Grid Plane for FC Model########

python src/plane.py \
	--experiment_name 'visualization' \
	--dataset_name 'CIFAR10' \
	--batch_size 128 \
	--model_name 'FC' \
	--model 'FCModel' \
	--input_dim 784 \
	--hidden_dims 1024 512 256 \
	--output_dim 10 \
	--seed "43" \
	--gpu_ids '0' \
	--init_start "/home/tdieudonne/dl3/src/checkpoints/model_A/final_model.pth" \
	--init_end "/home/tdieudonne/dl3/src/checkpoints/model_B/final_model.pth" \
	--fused_model_path "/home/tdieudonne/dl3/src/checkpoints/final_curve_fusion_model.pth" \
	--curve_ckpt "/home/tdieudonne/dl3/src/checkpoints/final_curve_model.pth" \
	--num_bends 3 \
	--curve="Bezier" \
    --grid_points 21 \
    --margin_left 0.25 \
    --margin_right 0.25 \
    --margin_top 0.25 \
    --margin_bottom 0.25
