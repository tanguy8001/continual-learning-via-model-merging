# Loss Landscape Visualization using PCA

This module provides tools to visualize neural network loss landscapes using Principal Component Analysis (PCA). It computes the principal components from multiple model checkpoints and projects the loss surface onto the 2D plane spanned by the first two principal components.

## Features

- **PCA-based Direction Computation**: Automatically computes the two most important directions in weight space using PCA
- **2D Contour Plots**: Visualize loss contours on the PCA plane
- **3D Surface Plots**: Interactive 3D visualization of the loss landscape
- **Training Trajectory**: Plot the optimization path of training checkpoints
- **Multiple Model Support**: Works with FC and MlpNet architectures
- **Caching**: Saves computed PCA directions and loss surfaces for reuse

## Requirements

```bash
pip install torch torchvision matplotlib seaborn scikit-learn h5py numpy
```

## Quick Start

### 1. Using the Command Line Interface

```bash
# Basic usage with existing model checkpoints
python loss_landscape_visualisation.py --model_folder checkpoints/ --model_name FC --dataset_name MNIST

# Custom visualization parameters
python loss_landscape_visualisation.py \
    --model_folder checkpoints/ \
    --model_name FC \
    --dataset_name MNIST \
    --x_range -1 1 \
    --y_range -1 1 \
    --resolution 51 \
    --save_dir ./my_results
```

### 2. Using the Example Script

```bash
# Run the example that creates sample models and visualizes them
python example_loss_landscape.py
```

### 3. Using the Python API

```python
from loss_landscape_visualisation import LossLandscapeVisualizer

# Create visualizer
visualizer = LossLandscapeVisualizer(
    model_name='FC',
    dataset_name='MNIST',
    device='cuda'  # or 'cpu'
)

# Run complete visualization pipeline
visualizer.visualize_loss_landscape(
    model_folder='./checkpoints',
    data_path='./data',
    x_range=(-1, 1),
    y_range=(-1, 1),
    resolution=51,
    save_dir='./results',
    show=False
)
```

## Command Line Arguments

| Argument         | Type         | Default                    | Description                                  |
| ---------------- | ------------ | -------------------------- | -------------------------------------------- |
| `--model_folder` | str          | **required**               | Folder containing model checkpoints          |
| `--model_name`   | str          | 'FC'                       | Model architecture ('FC', 'MlpNet')          |
| `--dataset_name` | str          | 'MNIST'                    | Dataset name ('MNIST')                       |
| `--data_path`    | str          | './data'                   | Path to the dataset                          |
| `--x_range`      | float, float | [-1, 1]                    | Range for x-coordinate (min max)             |
| `--y_range`      | float, float | [-1, 1]                    | Range for y-coordinate (min max)             |
| `--resolution`   | int          | 51                         | Resolution of the loss surface               |
| `--save_dir`     | str          | './loss_landscape_results' | Directory to save results                    |
| `--show`         | flag         | False                      | Display plots (not recommended for headless) |
| `--device`       | str          | 'auto'                     | Device to use ('cpu', 'cuda', 'auto')        |

## Output Files

The visualization creates the following directory structure:

```
save_dir/
├── pca_directions/
│   └── pca_directions.h5          # PCA directions and statistics
├── loss_surface/
│   └── loss_surface.h5            # Computed loss surface data
└── plots/
    ├── loss_contour_FC_MNIST.png  # 2D contour plot
    ├── loss_3d_FC_MNIST.png       # 3D surface plot
    └── trajectory_FC_MNIST.png    # Training trajectory plot
```

## Understanding the Results

### PCA Directions

- **1st Principal Component**: Captures the direction of maximum variance in the weight space
- **2nd Principal Component**: Captures the second most important direction (orthogonal to the first)
- **Explained Variance Ratio**: Shows how much variance each component explains

### Loss Landscape Interpretation

- **Flat regions**: Indicate stable areas where small weight changes don't affect loss much
- **Steep regions**: Indicate sensitive areas where small changes cause large loss changes
- **Valleys**: Represent good minima where the model performs well
- **Ridges**: Represent saddle points or poor minima

### Training Trajectory

- **Start point**: Initial model weights
- **End point**: Final trained model weights
- **Path**: Shows how the optimizer navigated the loss landscape

## Model Requirements

Your model checkpoints should be saved in the following format:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.get_model_config(),
    'val_acc': validation_accuracy,
    'test_acc': test_accuracy,
    'epoch': epoch_number
}, checkpoint_path)
```

## Supported Models

Currently supported model architectures:

- **FC**: Fully connected networks (FCModel.base)
- **MlpNet**: Multi-layer perceptron networks (MlpNetBase)

## Supported Datasets

Currently supported datasets:

- **MNIST**: Handwritten digit recognition

## Performance Tips

1. **Lower resolution for testing**: Use `--resolution 21` for faster computation
2. **Smaller ranges**: Use `--x_range -0.5 0.5` and `--y_range -0.5 0.5` for focused visualization
3. **GPU acceleration**: Use `--device cuda` if available
4. **Caching**: The script automatically caches PCA directions and loss surfaces

## Troubleshooting

### Common Issues

1. **"Need at least 2 model checkpoints"**

   - Ensure your model folder contains at least 2 `.pth` or `.checkpoint` files

2. **"Model checkpoint not found"**

   - Check that the model folder path is correct
   - Ensure checkpoint files have the correct format

3. **"Unsupported model"**

   - Currently only supports 'FC' and 'MlpNet' architectures
   - Check that your model inherits from the supported base classes

4. **Memory issues**
   - Reduce resolution: `--resolution 21`
   - Use smaller ranges: `--x_range -0.5 0.5`
   - Use CPU: `--device cpu`

### Debug Mode

For detailed error information, the script includes comprehensive error handling and will print stack traces when errors occur.

## Example Use Cases

### 1. Analyzing Training Stability

```bash
# Compare models trained with different learning rates
python loss_landscape_visualisation.py \
    --model_folder checkpoints_different_lr/ \
    --model_name FC \
    --dataset_name MNIST
```

### 2. Model Comparison

```bash
# Compare different model architectures
python loss_landscape_visualisation.py \
    --model_folder checkpoints_model_comparison/ \
    --model_name MlpNet \
    --dataset_name MNIST
```

### 3. Hyperparameter Analysis

```bash
# Analyze the effect of different hyperparameters
python loss_landscape_visualisation.py \
    --model_folder checkpoints_hyperparams/ \
    --model_name FC \
    --dataset_name MNIST \
    --resolution 71  # Higher resolution for detailed analysis
```

## Contributing

To add support for new models or datasets:

1. Ensure your model has a `get_model_config()` method
2. Add the model class to the `load_model()` method in `LossLandscapeVisualizer`
3. Add dataset support in `setup_data_loaders()`
4. Update the command line argument choices

## References

This implementation is inspired by:

- Li et al. "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
- The original loss landscape visualization codebase

## License

This code is part of the continual learning via model merging project.
