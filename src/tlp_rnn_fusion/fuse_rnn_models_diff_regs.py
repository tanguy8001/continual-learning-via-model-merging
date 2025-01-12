import numpy as np
import torch
import os
import logging
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from src.tlp_rnn_fusion import fuse_rnn_models
import argparse
from src.tlp_model_fusion import init




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='RNN')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--result_name', type=str, default='')

    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)

    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--hidden_dims', type=str, default=None)
    parser.add_argument('--output_dim', type=int, default=None)
    parser.add_argument('--hidden_activations', type=str, default=None)
    parser.add_argument('--encoder', default=False, action='store_true')

    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)

    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument("--seed", default=24601, type=int)

    parser.add_argument('--model_path_list', type=str, default=None, nargs='+',
                        help="Comma separated list of models and checkpoints"
                             "to be used fused together")

    # Fusion parameters
    parser.add_argument('--fusion_type', type=str, default=None,
                        choices=['tlp', 'tlp_no_hidden', 'avg', 'ot', 'fw'])
    parser.add_argument('--activation_batch_size', type=int, default=100)
    parser.add_argument('--use_pre_activations', default=False, action='store_true')
    parser.add_argument('--model_weights', default=None, type=float, nargs='+',
                        help='Comma separated list of weights for each model in fusion')

    parser.add_argument('--tlp_cost_choice', type=str, default='weight',
                        choices=['weight', 'activation'])
    parser.add_argument('--tlp_ot_solver', type=str, default='sinkhorn',
                        choices=['sinkhorn', 'emd'])
    parser.add_argument('--tlp_sinkhorn_regularization', type=float, default=0.001)
    parser.add_argument('--tlp_init_type', type=str, default=None,
                        choices=[None, 'identity', 'distill'])
    parser.add_argument('--tlp_init_model', type=int, default=None)

    parser.add_argument('--ad_hoc_cost_choice', type=str, default='weight',
                        choices=['weight', 'activation'])
    parser.add_argument('--ad_hoc_ot_solver', type=str, default='sinkhorn',
                        choices=['sinkhorn', 'emd'])
    parser.add_argument('--ad_hoc_sinkhorn_regularization', type=float, default=0.001)
    parser.add_argument('--ad_hoc_init_type', type=str, default=None,
                        choices=[None, 'distill'])
    parser.add_argument('--ad_hoc_initialization', type=int, default=None)

    # parser.add_argument('--fw_cost_choice', type=str, default=None)
    parser.add_argument('--fw_sinkhorn_regularization', type=float, default=0.01)
    parser.add_argument('--fw_single_layer_fusion_type', type=str, default=None,
                        choices=[None, 'support'])
    # parser.add_argument('--fw_minimization_type', type=str, default=None,
    #                     choices=[None, 'reg', 'pgd'],
                        # help="The type of minimization - None, with regularization or PGD")
    
    args = parser.parse_args(sys.argv[1:])

    args.hidden_dims = [int(s) for s in args.hidden_dims.strip('[]').split(',')]
    args.hidden_activations = None if args.hidden_activations is None else args.hidden_activations.strip('[]').split(',')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(level=logging.INFO)

    run_params = ['fusion', args.fusion_type,
                  'num_models', len(args.model_path_list),
                  'layers', len(args.hidden_dims) + 1, # for RNN with hidden layers and one output linear layer
                  'seed', args.seed]

    if args.fusion_type == 'tlp':
        run_params.extend(['cost_choice', args.tlp_cost_choice,
                           'solver', args.tlp_ot_solver])
        if args.use_pre_activations:
            run_params.extend(['preact'])
        if args.tlp_init_type is not None:
            run_params.extend(['init', args.tlp_init_type,
                               'model', args.tlp_init_model])
        if args.tlp_ot_solver == 'sinkhorn':
            run_params.extend(['reg', args.tlp_sinkhorn_regularization])
    elif args.fusion_type == 'ot':
        run_params.extend(['cost_choice', args.ad_hoc_cost_choice,
                           'solver', args.ad_hoc_ot_solver])
        if args.use_pre_activations:
            run_params.extend(['preact'])
        if args.ad_hoc_initialization is not None:
            run_params.extend(['init', args.ad_hoc_initialization])
        if args.ad_hoc_ot_solver == 'sinkhorn':
            run_params.extend(['reg', args.ad_hoc_sinkhorn_regularization])
    elif args.fusion_type == 'fw':
        run_params.extend(['reg', args.fw_sinkhorn_regularization,
                           'fusion', str(args.fw_single_layer_fusion_type)])

    run_str = '_'.join([str(x) for x in run_params])

    # run experiments with different regularizaton parameters
    # regularizations = np.logspace(-3,0,50)
    # regularizations = np.arange(0,1,0.02)
    regularizations = np.array([0.001,0.005,0.01,0.05,0.1,0.5])
    bse_result_name = args.result_name
    for reg in regularizations:
        print("Current regularization param is:",reg)

        args.tlp_sinkhorn_regularization = reg
        args.result_name = bse_result_name + "_reg" + str(reg) + ".pth"

        train_init = init.Init(args=args, run_str=run_str)
        fuse_models = fuse_rnn_models.FuseModels(args, train_init)
        fuse_models.fuse()
        

if __name__ == "__main__":
    print("torch.cuda.is_available()",torch.cuda.is_available())
    main()
