python src/test_curve_merging_mlp.py \
 --curve.interpolation-type STATIC \
 --curve.interpolation-points 0.0 1.0 \
 --curve.hidden-dim 64 \
 --curve.learning-rate 0.01 \
 --curve.momentum 0.9 \
 --curve.epochs 10 \
 --model.batch-size 128 \
 --model.num-workers 4 \
 --model.model-epochs 10 \
 --model.input-dim 3072 \
 --model.output-dim 10 \
 --model.hidden-dims 400 200 100 \
 --buffer.percentage 0.4

-----------------
In the cluster:
    sbatch --gpus=1 --mem-per-cpu=64g experiment_buffer_size.sh