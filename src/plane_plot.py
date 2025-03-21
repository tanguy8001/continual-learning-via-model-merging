# -*- coding: utf-8 -*-
"""
DO NOT USE: current version not stable, to be updated.
Plot the compute plane computed in plane.py
"""

# Commented out IPython magic to ensure Python compatibility.
### Visualizing grid plane
import argparse
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from torch.utils.data import Subset, ConcatDataset, DataLoader
from data import double_loaders
from plane import load_model
import curves
import models as mods
from models import mlpnet, fcmodel
from curve_merging import CurveConfig
import fuse_models
from models import mlpnet, fcmodel
import torch

# %matplotlib inline

parser = argparse.ArgumentParser(description='Plane visualization')
parser.add_argument('--result_path', type=str, default='result')
parser.add_argument('--experiment_name', type=str, default='visualization')
parser.add_argument('--model_name', type=str, default='FC')
parser.add_argument('--dataset_name', type=str, default='MNIST')
parser.add_argument('--curve_ckpt', type=str, default=None)
parser.add_argument('--curve', type=str, default="Bezier", metavar='CURVE',
                    help='curve type to use (default: Bezier)')
parser.add_argument('--curve_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--model', type=str, default='FCModel')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--tr_vmax', type=float, default=0.4,
                    help='color normalization parameter vmax for training loss visualization')
parser.add_argument('--tr_log_alpha', type=float, default=-5.0,
                    help='color normalization parameter log_alpha for training loss visualization')
parser.add_argument('--te_vmax', type=float, default=8.0,
                    help='color normalization parameter vmax for test error visualization')
parser.add_argument('--te_log_alpha', type=float, default=-5.0,
                    help='color normalization parameter log_alpha for test error visualization')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_path = os.path.join(args.result_path, args.experiment_name + '_' + args.model_name + '_' + args.dataset_name)

file = np.load(os.path.join(output_path, 'plane.npz'))

matplotlib.rc('text', usetex=False)
#matplotlib.rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath'])
matplotlib.rc('text.latex', preamble=r'\usepackage{sansmath}\n\sansmath')
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})

matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

sns.set_style('whitegrid')

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

def create_fused_loader(loader_A, loader_B, batch_size, num_workers):
    """Create a fused data loader from two separate loaders."""
    # Get underlying datasets
    dataset_A = loader_A.dataset
    dataset_B = loader_B.dataset
    
    # Combine datasets
    fused_dataset = ConcatDataset([dataset_A, dataset_B])
    
    # Create new loader with combined data
    fused_loader = DataLoader(
        fused_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return fused_loader

class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)

def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
  cmap = plt.get_cmap(cmap)
  if vmax is None:
    clipped = values.copy()
  else:
    clipped = np.minimum(values, vmax)
  log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
  levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
  levels[0] = clipped.min()
  levels[-1] = clipped.max()
  levels = np.concatenate((levels, [1e10]))
  norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)

  contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                        linewidths=2.5,
                        zorder=1,
                        levels=levels)
  contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                          levels=levels,
                          zorder=0,
                          alpha=0.55)
  colorbar = plt.colorbar(format='%.2g')
  labels = list(colorbar.ax.get_yticklabels())
  labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
  colorbar.ax.set_yticklabels(labels)
  return contour, contourf, colorbar

plt.figure(figsize=(12.4, 7))

contour, contourf, colorbar = plane(
    file['grid'],
    file['tr_loss'],
    vmax = args.tr_vmax,
    log_alpha = args.tr_log_alpha,
    N = 7
)

# Parameters
batch_size = 128
num_workers = 4
input_dim = 28 * 28  # MNIST image size
hidden_dims = [800, 400, 200]
num_classes = 10
test_digit = 4  # The digit to split on

# Create data directory if it doesn't exist
data_path = os.path.join(os.getcwd(), "data")
os.makedirs(data_path, exist_ok=True)

data_loaders, num_classes = double_loaders(
    dataset="MNIST",
    path=data_path,
    batch_size=batch_size,
    num_workers=num_workers,
    transform_name="MLPNET",
    digit=test_digit
)

# Create fused loader for curve training
fused_loader = create_fused_loader(
    data_loaders['trainA'],
    data_loaders['trainB'],
    batch_size=batch_size,
    num_workers=num_workers
)

    
config = CurveConfig()
architecture = getattr(mods, args.model)
architecture.kwargs['input_dim'] = config.input_dim
architecture.kwargs['hidden_dims'] = config.hidden_dims
architecture.kwargs['output_dim'] = config.output_dim
#architecture.kwargs['num_classes'] = 10
curve = getattr(curves, args.curve)
curve_model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve, # MlpNetCurve
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
curve_model.to(device)
state_dict = torch.load(args.curve_ckpt)
curve_model.load_state_dict(state_dict['model_state_dict'])

w = list()
curve_parameters = list(curve_model.net.parameters())   # List of tensors each representing a learnable param
# Iterate over number of bends that each correspond to a separate set of weights in the curve model
for i in range(args.num_bends):
    # Extract and organize weights for each bend
    w.append(np.concatenate([
        p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
    ]))

ts = np.linspace(0.0, 1.0, args.curve_points)
curve_coordinates = []
# Weights along the curve sampled at equally spaced t values, then projected
for t in np.linspace(0.0, 1.0, args.curve_points):
    weights = curve_model.weights(torch.Tensor([t]).to(device))
    # weights.shape == w.shape
    curve_coordinates.append(get_xy(weights, w[0], u, v))
curve_coordinates = np.stack(curve_coordinates)


bend_coordinates = file['bend_coordinates']
fused_model_coordinates = file['fused_model_coordinates']

plt.scatter(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], marker='o', c='k', s=120, zorder=2)
#plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='D', c='k', s=120, zorder=2)
plt.scatter(fused_model_coordinates[0], fused_model_coordinates[1], marker='D', c='r', s=120, zorder=2)
plt.plot(curve_coordinates[:, 0], curve_coordinates[:, 1], linewidth=4, c='k', label='$w(t)$', zorder=4)
plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', 
         dashes=(3, 4), linewidth=3, zorder=2)
#plt.plot(bend_coordinates[[0, 1], 0], bend_coordinates[[0, 1], 1], c='k', linestyle='--', 
#         dashes=(3, 4), linewidth=3, zorder=2)

plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
plt.savefig(os.path.join(output_path, 'train_loss_plane.png'), format='png', bbox_inches='tight')
#plt.show()

plt.figure(figsize=(12.4, 7))

contour, contourf, colorbar = plane(
    file['grid'],
    file['te_err'],
    vmax = args.te_vmax,
    log_alpha = args.te_log_alpha,
    N = 7
)

bend_coordinates = file['bend_coordinates']
fused_model_coordinates = file['fused_model_coordinates']

plt.scatter(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], marker='o', c='k', s=120, zorder=2)
#plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='D', c='k', s=120, zorder=2)
plt.scatter(fused_model_coordinates[0], fused_model_coordinates[1], marker='D', c='r', s=120, zorder=2)
plt.plot(curve_coordinates[:, 0], curve_coordinates[:, 1], linewidth=4, c='k', label='$w(t)$', zorder=4)
plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', 
         dashes=(3, 4), linewidth=3, zorder=2)
#plt.plot(bend_coordinates[[0, 1], 0], bend_coordinates[[0, 1], 1], c='k', linestyle='--', 
#         dashes=(3, 4), linewidth=3, zorder=2)

plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
plt.savefig(os.path.join(output_path, 'test_error_plane.png'), format='png', bbox_inches='tight')
#plt.show()