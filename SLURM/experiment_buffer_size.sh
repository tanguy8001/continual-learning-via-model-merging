#!/bin/bash
#SBATCH --mem-per-cpu=20G            # CPU memory PER CPU CORE allocated to the task
#SBATCH --gpus=1                     # Request 1 GPU PER TASK
#SBATCH --gres=gpumem:5g             # Request 5GB of GPU memory PER TASK (added 'g' for clarity, Slurm often infers it)

module load stack/2024-06  gcc/12.2.0 python/3.11.6 eth_proxy cuda/12.8.0

#export WANDB_API_KEY=59ca4051dcb31d2f1df3b5e6c98a23e6af863a27

#source $HOME/my_venv/bin/activate   # Tyro is already installed here


#python3 /cluster/home/lbarinka/continual-learning-via-model-merging/src/test_curve_merging_mlp.py \
 #--curve.interpolation-type DYNAMIC \
 #--curve.interpolation-points 0.0 1.0 \
 #--curve.hidden-dim 64 \
 #--curve.learning-rate 0.01 \
 #--curve.momentum 0.9 \
 #--curve.epochs 10 \
 #--model.batch-size 128 \
 #--model.num-workers 4 \
 #--model.model-epochs 10 \
 #--model.input-dim 3072 \
 #--model.output-dim 10 \
 #--model.hidden-dims 400 200 100 \
 #--buffer.percentage 0.01 



#python3 /cluster/home/lbarinka/continual-learning-via-model-merging/src/test_curve_merging_mlp.py \
 #--curve.interpolation-type STATIC \
 #--curve.interpolation-points 0.0 1.0 \
 #--curve.hidden-dim 64 \
 #--curve.learning-rate 0.01 \
 #--curve.momentum 0.9 \
 #--curve.epochs 10 \
 #--model.batch-size 128 \
 #--model.num-workers 4 \
 #--model.model-epochs 10 \
 #--model.input-dim 3072 \
 #--model.output-dim 10 \
 #--model.hidden-dims 400 200 100 \
 #--buffer.percentage 0.8


 #python3 /cluster/home/lbarinka/continual-learning-via-model-merging/src/test_curve_merging_mlp.py\
 #--curve.interpolation-type STATIC \
 #--curve.interpolation-points 0.0 1.0 \
 #--curve.hidden-dim 64 \
 #--curve.learning-rate 0.01 \
 #--curve.momentum 0.9 \
 #--curve.epochs 10 \
 #--model.batch-size 128 \
 #--model.num-workers 4 \
 #--model.model-epochs 10 \
 #--model.input-dim 3072 \
 #--model.output-dim 10 \
 #--model.hidden-dims 400 200 100 \
 #--buffer.percentage 0.6