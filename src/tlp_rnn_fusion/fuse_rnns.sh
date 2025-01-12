 fusion_type="tlp"
 tlp_cost_choice="weight"
 python fuse_rnn_models.py \
               --experiment_name 'rnn_fusion_tlp' \
               --dataset_name 'names' \
               --model_name 'RNN' \
               --input_dim 57 \
               --hidden_dims [30] \
               --output_dim 18 \
               --hidden_activations ['tanh'] \
               --eval_batch_size 3 \
               --num_epochs 20 \
               --seed "43" \
               --gpu_ids "1" \
               --fusion_type "$fusion_type" \
               --activation_batch_size 256 \
               --use_pre_activations \
               --tlp_cost_choice "${tlp_cost_choice}" \
               --tlp_init_type 'identity' \
               --tlp_init_model 0 \
               --tlp_ot_solver 'sinkhorn' \
               --tlp_sinkhorn_regularization 0.001 \
               --model_path_list \
               "RNN,trained_models/names_1.pth" \
               "RNN,trained_models/names_2.pth"


