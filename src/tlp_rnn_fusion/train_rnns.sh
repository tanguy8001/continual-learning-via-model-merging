# initialize a new model 
# RNN dims: 57 + [30] + 18
# --model_save_path 'trained_models/names_1.pth'
python ./train_rnn.py \
            --task_name 'name_classification' \
            --model_name 'rnn' \
            --dataset_name 'names' \
            --num_epochs 2 \
            --batch_size 5 \
            --gpu_batch_size 20 \
            --device 'cpu' \
            --save_models  \
            --save_models_every 10 \
            --model_save_path './trained_models/names_1.pth' \
            --input_dim 57 \
            --hidden_dims [30] \
            --output_dim 18 \
            --include_date

# # load a trained model and resume training
# # RNN dims: 57 + [30] + 18
# # --model_save_path './trained_models/names_1.pth' \
# python ./train_rnn.py \
#             --task_name 'name_classification' \
#             --model_name 'rnn' \
#             --dataset_name 'names' \
#             --num_epochs 2 \
#             --batch_size 5 \
#             --gpu_batch_size 20 \
#             --device 'cpu' \
#             --save_models  \
#             --save_models_every 10 \
#             --pretrained  \
#             --model_save_path './trained_models/names_1.pth' \
#             --model_load_path 'trained_models/names_1.pth' \
#             --input_dim 57 \
#             --hidden_dims [30] \
#             --output_dim 18 \
#             --include_date

# # initialize a new model 
# # RNN dims: 57 + [40] + 18
# # --model_save_path 'trained_models/names_2.pth'
# python ./train_rnn.py \
#             --task_name 'name_classification' \
#             --model_name 'rnn' \
#             --dataset_name 'names' \
#             --num_epochs 2 \
#             --batch_size 5 \
#             --gpu_batch_size 20 \
#             --device 'cpu' \
#             --save_models  \
#             --save_models_every 10 \
#             --model_save_path './trained_models/names_2.pth' \
#             --input_dim 57 \
#             --hidden_dims [40] \
#             --output_dim 18 \
#             --include_date

# load a trained model and resume training
# RNN dims: 57 + [40] + 18
# --model_save_path './trained_models/names_2.pth' \
python ./train_rnn.py \
            --task_name 'name_classification' \
            --model_name 'rnn' \
            --dataset_name 'names' \
            --num_epochs 2 \
            --batch_size 5 \
            --gpu_batch_size 20 \
            --device 'cpu' \
            --save_models  \
            --save_models_every 10 \
            --pretrained  \
            --model_save_path './trained_models/names_2.pth' \
            --model_load_path 'trained_models/names_2.pth' \
            --input_dim 57 \
            --hidden_dims [40] \
            --output_dim 18 \
            --include_date


# all sys arguments
# python ./train_rnn.py \
#             --task_name name_classification \
#             --model_name rnn \
#             --dataset_name names \
#             --num_epochs 2 \
#             --steps_per_iter 100 \
#             --test_steps_per_iter 25 \
#             --learning_rate 1e-3 \
#             --batch_size 5 \
#             --gpu_batch_size 20 \
#             --device cpu \
#             --log_to_wandb  \
#             --note '' \
#             --wandb_project my_project \
#             --include_date  \
#             --save_models  \
#             --save_models_every 10 \
#             --pretrained  \
#             --model_load_path '' \
            # --model_save_path './trained_models/names_1.pth' \
#             --input_dim 57 \
#             --hidden_dims=[30] \
#             --output_dim 18 \
#             --hidden_activations=['tanh']\
#             --bias  \


# RNN model with encoders and decoder, trained on shakespeare next character prediction
python ./train_rnn_new.py \
            --model_name 'rnn' \
            --dataset_name 'names' \
            --train_data_path_x '' \
            --test_data_path_x '' \
            --train_data_path_y '' \
            --test_data_path_y '' \
            --num_epochs 2 \
            --learning_rate 0.001 \
            --batch_size 5 \
            --device 'cpu' \
            --model_load_path '' \
            --model_save_path './trained_models/names_1.pth' \
            --vocab_size 80 \
            --embed_dim \
            --hidden_dims [30] \
            --pretrained


python ./train_rnn_new.py \
            --model_name 'rnn' \
            --dataset_name 'shakespeare_char' \
            --train_data_path_x './shakespeare_char/10_splits_169600/train_x_0.txt' \
            --test_data_path_x './shakespeare_char/10_splits_169600/test_x.txt' \
            --train_data_path_y './shakespeare_char/10_splits_169600/train_y_0.txt' \
            --test_data_path_y './shakespeare_char/10_splits_169600/test_y.txt' \
            --num_epochs 1 \
            --learning_rate 0.001 \
            --batch_size 100 --device 'cpu' \
            --model_load_path '' \
            --model_save_path './trained_models/names_1.pth' \
            --vocab_size 80 \
            --embed_dim 50 \
            --hidden_dims [30,50]\
            --momentum 0.9 \
            --optimizer 'SGD' \
            --weight_decay= .0001
            
            