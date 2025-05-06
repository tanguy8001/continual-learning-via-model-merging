python plot_loss_surface.py --dataset "MNIST" --batch_size 128 --model_a_path "checkpoints/mnist_model_A.pth" --model_b_path "checkpoints/mnist_model_B.pth" --curve_mlp_path "checkpoints/mnist_mlp_curve.pth"  --curve_net_ckpt "checkpoints/mnist_bezier_curve.pth" --output_plot_path "mnist_curvemlp_loss_surface.png" --input_dim 784 --hidden_dims 400 200 100 --output_dim 10 --gpu_ids "0"

# python src/plot_loss_surface.py --dataset "MNIST" --batch_size 128 --model_a_path "src/checkpoints/mnist_model_A.pth" --model_b_path "src/checkpoints/mnist_model_B.pth" --curve_mlp_path "src/checkpoints/mnist_mlp_curve.pth"  --curve_net_ckpt "src/checkpoints/mnist_bezier_curve.pth" --output_plot_path "mnist_curvemlp_loss_surface.png" --input_dim 784 --hidden_dims 400 200 100 --output_dim 10 --gpu_ids "0"

