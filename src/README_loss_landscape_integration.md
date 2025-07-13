# Loss Landscape Visualization Integration Guide

This guide explains how to add loss landscape visualization using PCA components from the learned curve to your existing curve merging setup.

## Overview

The loss landscape visualization uses Principal Component Analysis (PCA) to extract the most important directions from the learned curve between two models, then projects the loss surface onto the 2D plane spanned by these components.

## Key Components

1. **PCA from Learned Curve**: Instead of using random directions or training trajectory PCA, we use the curve learned by the CurveMLP to find the most important directions in weight space.

2. **Loss Surface Projection**: The loss landscape is computed on the 2D plane spanned by the first two principal components of the curve.

3. **Curve Trajectory Visualization**: The learned curve path is overlaid on the loss landscape to show how the curve navigates through the loss surface.

## Quick Start

### Option 1: Run the Demo Script

```bash
# First, run your existing curve merging to create models
python test_curve_merging_mlp.py

# Then run the loss landscape visualization
python run_loss_landscape.py
```

### Option 2: Integrate with Your Existing Test

Add these imports to your `test_curve_merging_mlp.py`:

```python
from run_loss_landscape import create_loss_landscape, plot_loss_landscape_with_curve
```

Then add this code at the end of your `main()` function, after the curve training:

```python
# Add loss landscape visualization
print("\nCreating loss landscape visualization...")
X, Y, loss_matrix, pca_components, explained_variance = create_loss_landscape(
    curve_mlp, model_A, model_B, test_loader, device,
    x_range=(-0.3, 0.3),
    y_range=(-0.3, 0.3),
    resolution=15  # Lower resolution for faster computation
)

plot_loss_landscape_with_curve(X, Y, loss_matrix, curve_mlp, model_A, model_B, pca_components)

print(f"\nLoss landscape analysis complete!")
print(f"1st PC explains {explained_variance[0]*100:.1f}% of variance")
print(f"2nd PC explains {explained_variance[1]*100:.1f}% of variance")
print(f"Total explained by 2 components: {np.sum(explained_variance[:2])*100:.1f}%")
```

## Understanding the Results

### PCA Components from Curve

- **1st Principal Component**: Captures the direction of maximum variance along the learned curve
- **2nd Principal Component**: Captures the second most important direction (orthogonal to the first)
- **Explained Variance**: Shows how much of the curve's variance each component explains

### Loss Landscape Interpretation

- **Flat regions**: Areas where small movements along the PCA directions don't affect loss much
- **Steep regions**: Sensitive areas where small changes cause large loss changes
- **Valleys**: Good minima where the model performs well
- **Ridges**: Saddle points or poor minima

### Curve Trajectory

- **Red line**: The learned curve path through the loss landscape
- **Green dot**: Model A (t=0)
- **Blue dot**: Model B (t=1)
- **Magenta dot**: Midpoint (t=0.5)

## Key Functions

### `compute_pca_from_curve(curve_mlp, model_A, model_B, num_points=500)`

Computes PCA components from the learned curve by:

1. Sampling points along the curve using the CurveMLP
2. Computing PCA on the sampled weight vectors
3. Returning the principal components and explained variance ratios

### `create_loss_landscape(curve_mlp, model_A, model_B, test_loader, device, x_range, y_range, resolution)`

Creates a loss landscape grid by:

1. Computing PCA components from the curve
2. Creating a grid in PCA space
3. Computing loss at each grid point
4. Returning the coordinates, loss matrix, and PCA information

### `plot_loss_landscape_with_curve(X, Y, loss_matrix, curve_mlp, model_A, model_B, pca_components)`

Creates a visualization showing:

1. Contour plot of the loss landscape
2. The learned curve trajectory overlaid
3. Key points along the curve (start, end, midpoint)

## Performance Considerations

- **Resolution**: Lower resolution (15x15) for faster computation, higher (51x51) for detailed visualization
- **Range**: Smaller ranges (-0.3 to 0.3) focus on the curve area, larger ranges show broader context
- **Num points**: More points (1000) for accurate PCA, fewer (500) for faster computation

## Example Output

The visualization will show:

1. A contour plot of the loss landscape in PCA space
2. The learned curve trajectory (red line)
3. Key points: Model A (green), Model B (blue), midpoint (magenta)
4. Explained variance percentages in the title

## Integration with WandB

You can log the PCA results to WandB:

```python
wandb.log({
    "pca_explained_variance_pc1": explained_variance[0],
    "pca_explained_variance_pc2": explained_variance[1],
    "pca_total_explained_variance": np.sum(explained_variance[:2])
})
```

## Troubleshooting

1. **Models not found**: Make sure to run `test_curve_merging_mlp.py` first to create the model checkpoints
2. **Memory issues**: Reduce resolution or use smaller ranges
3. **Slow computation**: Use lower resolution or fewer PCA sampling points
4. **Plot not showing**: Make sure matplotlib backend is properly configured

## Advanced Usage

For more detailed analysis, you can:

1. **Compare with random directions**: Use random PCA directions as a baseline
2. **Analyze curve smoothness**: Look at how smooth the curve trajectory is
3. **Study loss valleys**: Identify good interpolation paths
4. **Optimize curve parameters**: Use the visualization to tune curve MLP hyperparameters

## Files Created

- `curve_loss_landscape.png`: The main visualization
- `curve_dimensionality.png`: PCA dimensionality analysis (if using the full script)

This integration provides valuable insights into how the learned curve navigates the loss landscape and can help optimize the curve merging process.
