#!/bin/bash
#SBATCH --mem-per-cpu=20G            
#SBATCH --gpus=1                     
#SBATCH --gres=gpumem:5g      

module load stack/2024-06  gcc/12.2.0 python/3.11.6 eth_proxy cuda/12.8.0
source /cluster/home/lbarinka/continual-learning-via-model-merging/.venv/bin/activate   

export WANDB_API_KEY=59ca4051dcb31d2f1df3b5e6c98a23e6af863a27

python3 /cluster/home/lbarinka/continual-learning-via-model-merging/src/test_curve_merging_mlp.py \
--curve.interpolation-type DYNAMIC \
--curve.interpolation-points 0.0 1.0 \
--curve.hidden-dim 64 \
--curve.learning-rate 0.07 \
--curve.momentum 0.9 \
--curve.epochs 10 \
--model.batch-size 128 \
--model.num-workers 4 \
--model.model-epochs 10 \
--model.input-dim 784 \
--model.output-dim 10 \
--model.hidden-dims 400 200 100 \
--buffer.percentage 0.1

# Then run the loss landscape visualization
python3 /cluster/home/lbarinka/continual-learning-via-model-merging/src/run_loss_landscape.py 