#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=logs/curve_merging_mlp_%j.out

cd /work/courses/3dv/24/clmm
source /home/tdieudonne/.bashrc
conda activate slam

echo "Starting curve merging with MLP at: $(date)"
export PYTHONPATH=$PYTHONPATH:.

#### Curve merging with MLP #####
 python src/test_curve_merging_mlp.py \

