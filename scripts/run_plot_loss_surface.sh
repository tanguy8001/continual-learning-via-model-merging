python src/plot_loss_surface.py \
	--experiment_name 'mnist_visualization' \
	--dataset 'MNIST' \
	--batch_size 128 \
	--model_a_path "src/checkpoints/mnist_model_A.pth" \
	--model_b_path "src/checkpoints/mnist_model_B.pth" \
	--curve_mlp_path "src/checkpoints/mnist_curve_mlp.pth" \
	--seed "43" \
	--gpu_ids '0' \
	--resolution 20 \
	--xmin -1.0 \
	--xmax 1.0 \
	--ymin -1.0 \
	--ymax 1.0 \
	--output_plot_path 'mnist_curvemlp_loss_surface.png'

# python src/plot_loss_surface.py --dataset "MNIST" --batch_size 128 --model_a_path "src/checkpoints/mnist_model_A.pth" --model_b_path "src/checkpoints/mnist_model_B.pth" --curve_mlp_path "src/checkpoints/mnist_curve_mlp.pth" --resolution 20 --xmin -1.0 --xmax 1.0 --ymin -1.0 --ymax 1.0 --output_plot_path "mnist_curvemlp_loss_surface.png" --input_dim 784 --hidden_dims 400 200 100 --output_dim 10 --gpu_ids "0"