# -*- coding: utf-8 -*-
"""
Computes the plane to be plotted with the two base models and their merged model.
"""

import torch
import numpy as np
import torch.nn.functional as F
import tqdm
import tabulate
import os
import argparse
import logging

from torch.utils import data
from torchvision import datasets, transforms

from utils import average_meter
import train_models
import fuse_models
from fuse_models import get_model
from init import make_dirs

import curves
import models as mods
from models import mlpnet, fcmodel
from curve_merging import CurveConfig

### Load neural networks
def load_model(model_name, model_path):
  print(model_path)
  state_dict = torch.load(model_path)
  model = fuse_models.get_model(model_name, state_dict['config'])
  model.load_state_dict(state_dict['model_state_dict'])
  return model 

### Get coordinate
def get_xy(point, origin, vector_x, vector_y):
  return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

##### Test function
def test(dataloader, model, model_name='FC'):
  #tbar = tqdm.tqdm(dataloader)
  total = 0
  correct = 0
  loss_logger = average_meter.AverageMeter()
  
  if torch.cuda.is_available():
    model = model.cuda()
  
  model.eval()

  for batch_idx, (images, labels) in enumerate(dataloader):
    if torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()
    if model_name == 'FC':
      logits = model(images.view(images.size(0), -1))
    elif model_name in ['ImageRNN', 'ImageLSTM', 'RNN']:
      logits = model(images.squeeze())
    else:
      logits = model(images)
    
    loss = F.cross_entropy(logits, labels)
    prediction = torch.argmax(logits, dim=1)
    total += images.size(0)
    correct += torch.sum(labels == prediction)
    loss_logger.update(loss.item())

  accuracy = 100.0 * correct / total
  return {
      'loss': loss_logger.avg,
      'accuracy': accuracy,
  }

def loss_func(last_output, y):
  m = torch.nn.LogSoftmax(dim=1)
  loss = torch.nn.NLLLoss(reduction='mean')

  return (loss(m(last_output), y))

    


### Get all the weight in one nerual network
def get_weight(model):
  weights = np.concatenate([p.cpu().detach().numpy().ravel() for p in model.parameters()])
  return weights

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str, default='test')
  parser.add_argument('--model_name', type=str, default='FC')
  parser.add_argument('--dataset_name', type=str, default='MNIST')
  parser.add_argument('--result_path', type=str, default='/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints')

  parser.add_argument('--data_path', type=str, default='./data')
  parser.add_argument('--batch_size', type=int, default=64)

  parser.add_argument('--normalize', default=False, action='store_true')
  parser.add_argument('--nsplits', type=int, default=1,
                        help='Number of splits of the dataset')
  parser.add_argument('--split_index', type=int, default=1,
                        help='The current index of split dataset used!')
  parser.add_argument('--ds_scale_factor', type=float, default=1.0,
                        help='To understand effect of ds scaling')
  parser.add_argument('--alpha_h', type=float, default=None, nargs='+',
                      help='The weight for the hidden to hidden matrix costs')

  parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
  parser.add_argument('--curve_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
  parser.add_argument('--curve_ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint of the trained curve (default: None)')
  parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
  parser.add_argument('--model', type=str, default='FCModel')
  parser.add_argument('--grid_points', type=int, default=21,
                      help='number of points in the grid (default: 21)')
  parser.add_argument('--margin_left', type=float, default=0.2,
                      help='left margin (default: 0.2)')
  parser.add_argument('--margin_right', type=float, default=0.2,
                      help='right margin (default: 0.2)')
  parser.add_argument('--margin_bottom', type=float, default=0.2,
                      help='bottom margin (default: 0.)')
  parser.add_argument('--margin_top', type=float, default=0.2,
                      help='top margin (default: 0.2)')

  parser.add_argument('--input_dim', type=int, default=784)
  parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
  parser.add_argument('--output_dim', type=int, default=10)

  parser.add_argument('--evaluate', default=False, action='store_true')
  parser.add_argument('--resume', default=False, action='store_true')
  parser.add_argument('--init_start', type=str, default=None,
                      help='checkpoint to init start point (default: None)')
  parser.add_argument('--init_end', type=str, default=None,
                      help='checkpoint to init end point (default: None)')
  parser.add_argument('--fused_model_path', type=str, default=None,
                      help='checkpoint to the fused model')
  parser.add_argument('--permuted_model_path', type=str, default=None)

  parser.add_argument('--no_cuda', default=False, action='store_true')
  parser.add_argument('--gpu_ids', type=str, default='0')
  parser.add_argument("--seed", default=24601, type=int)

  parser.add_argument('--hetero_special_digit', type=int, default=4,
                        help='Special digit for heterogeneous MNIST')
  parser.add_argument('--hetero_special_train', default=False, action='store_true',
                        help='If HeteroMNIST with special digit split is used for training.')
  parser.add_argument('--hetero_other_digits_train_split', default=0.9, type=float)
  parser.add_argument('--heterogeneous', default=False, action='store_true')

  parser.add_argument('--finetune_visualization', default=False, action='store_true')
  parser.add_argument('--finetuned_model_path', type=str, default=None,
                      help='checkpoint to the finetuned model') 
  

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  
  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if args.heterogeneous:
    model_name = args.experiment_name + '_' + args.model_name + '_' + 'HeteroMNIST'
  else:
    model_name = args.experiment_name + '_' + args.model_name + '_' + args.dataset_name 
  output_path = os.path.join(args.result_path, model_name)
  make_dirs(output_path)
  name = model_name
  logging.info("Generating grid plane for %s", name)


  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
  args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  logging.basicConfig(level=logging.INFO)

  if args.finetune_visualization:
    model1 = load_model(args.model_name, args.init_start)
    model2 = load_model(args.model_name, args.init_end)
    fused_model = load_model(args.model_name, args.fused_model_path)
    finetuned_model = load_model(args.model_name, args.finetuned_model_path)

    #w = [get_weight(model1), get_weight(finetuned_model), get_weight(fused_model)]
    #w = [get_weight(model1), get_weight(finetuned_model), get_weight(model2)]
    logging.info("Weight space dimentionality: {}".format(w[0].shape[0]))
  else:
    model1 = load_model(args.model_name, args.init_start)
    model2 = load_model(args.model_name, args.init_end)
    fused_model = load_model(args.model_name, args.fused_model_path)

    fused_model_w = get_weight(fused_model)
    #w = [get_weight(model1), get_weight(fused_model), get_weight(model2)]

  config = CurveConfig()
  architecture = getattr(mods, args.model)
  architecture.kwargs['input_dim'] = config.input_dim
  architecture.kwargs['hidden_dims'] = config.hidden_dims
  architecture.kwargs['output_dim'] = config.output_dim
  curve = getattr(curves, args.curve)

  curve_model = curves.CurveNet(
      config.num_classes,
      curve,
      architecture.curve, # FCModelCurve
      args.num_bends,
      architecture_kwargs=architecture.kwargs,
  )
  curve_model.to(device)
  checkpoint = torch.load(args.curve_ckpt)
  curve_model.load_state_dict(checkpoint['model_state_dict'])

  w = list()
  curve_parameters = list(curve_model.net.parameters())   # List of tensors each representing a learnable param
  # Iterate over number of bends that each correspond to a separate set of weights in the curve model
  for i in range(args.num_bends):
      # Extract and organize weights for each bend
      w.append(np.concatenate([
          p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
      ]))

  logging.info("Weight space dimentionality: {}".format(w[0].shape[0]))
  config = model1.get_model_config()

  ### Generate orthonormal basises
  u = w[2] - w[0]     # The two endpoints of the curve
  dx = np.linalg.norm(u)
  u /= dx

  v = w[1] - w[0]
  v -= np.dot(u, v) * u
  dy = np.linalg.norm(v)
  v /= dy

  bend_coordinates = np.stack([get_xy(p, w[0], u, v) for p in w])
  logging.info('The coordinates of model 1 on the plane: {}'.format(bend_coordinates[0]))
  logging.info('The coordinates of model 2 on the plane {}'.format(bend_coordinates[2]))

  fused_model_coordinates = get_xy(fused_model_w, w[0], u, v)
  logging.info('The coordinates of the fused model on the plane: {}'.format(fused_model_coordinates))

  ts = np.linspace(0.0, 1.0, args.curve_points)
  curve_coordinates = []
  # Weights along the curve sampled at equally spaced t values, then projected
  for t in np.linspace(0.0, 1.0, args.curve_points):
      weights = curve_model.weights(torch.Tensor([t]).to(device)) # TODO
      curve_coordinates.append(get_xy(weights, w[0], u, v))
  curve_coordinates = np.stack(curve_coordinates)

  ### Generate the grid plane
  trainloader, valoader, testloader = train_models.get_dataloaders(args)

  
  # Test
  logging.info('Test accuracy of model 1:{}'.format(test(testloader, model1, args.dataset_name)))
  logging.info('Test accuracy of model 2:{}'.format(test(testloader, model2, args.dataset_name)))
  logging.info('Test accuracy of fused model:{}'.format(test(testloader, fused_model, args.dataset_name)))

  G = args.grid_points
  alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
  betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

  tr_loss = np.zeros((G, G))
  tr_acc = np.zeros((G, G))
  tr_err = np.zeros((G, G))

  te_loss = np.zeros((G, G))
  te_acc = np.zeros((G, G))
  te_err = np.zeros((G, G))

  grid = np.zeros((G, G, 2))
  
  base_model = get_model(args.model_name, config)
  base_model.to(device)

  columns = ['X', 'Y', 'Train loss', 'Train error (%)', 'Test error (%)']
  logging.info("Begin to generate grid plane.")

  for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
      # Generate the weights of the neural networks at point (i, j)
      # => Corresponds to moving in the u-v directions from w0
      p = w[0] + alpha * dx * u + beta * dy * v   # Weight vector p = w0 ​+ α⋅dx⋅u + β⋅dy⋅v

      offset = 0
      for parameter in base_model.parameters():
        size = np.prod(parameter.size())
        value = p[offset:offset+size].reshape(parameter.size())     # Reshape p into format required by model
        parameter.data.copy_(torch.from_numpy(value))
        offset += size

      tr_res = test(trainloader, base_model, args.model_name)
      te_res = test(testloader, base_model, args.model_name)

      tr_loss_v, tr_acc_v = tr_res['loss'], tr_res['accuracy']
      te_loss_v, te_acc_v = te_res['loss'], te_res['accuracy']

      c = get_xy(p, w[0], u, v)
      grid[i, j] = [alpha * dx, beta * dy]

      tr_loss[i, j] = tr_loss_v
      tr_acc[i, j] = tr_acc_v
      tr_err[i, j] = 100.0 - tr_acc[i, j]

      te_loss[i, j] = te_loss_v
      te_acc[i, j] = te_acc_v
      te_err[i, j] = 100.0 - te_acc[i, j]

      values = [
              grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_err[i, j], te_err[i, j]
      ]

      table = tabulate.tabulate([values], columns, tablefmt = 'simple', floatfmt='10.4f')
      if j == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
      else:
        table = table.split('\n')[2]
      print(table)
  
  np.savez(
      os.path.join(output_path, 'plane.npz'),
      bend_coordinates = bend_coordinates,
      fused_model_coordinates = fused_model_coordinates,
      curve_coordinates = curve_coordinates,
      alphas = alphas,
      betas = betas,
      grid = grid,
      tr_loss = tr_loss,
      tr_acc = tr_acc,
      tr_err = tr_err,
      te_loss = te_loss,
      te_acc = te_acc,
      te_err = te_err
  )

if __name__ == "__main__":
  main()