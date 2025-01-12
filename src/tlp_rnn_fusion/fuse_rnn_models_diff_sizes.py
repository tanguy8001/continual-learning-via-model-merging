import argparse
import copy
import logging
import os
import pdb
import torch
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from src.tlp_model_fusion import ad_hoc_ot_fusion
from src.tlp_model_fusion import avg_fusion
from src.tlp_model_fusion import frank_wolfe_fusion
from src.tlp_model_fusion import init
from src.tlp_model_fusion import train_models

from src.tlp_rnn_fusion import rnn_models
from src.tlp_rnn_fusion import tlp_fusion_rnn

from tlp_rnn_fusion.rnn_models import RNNWithEncoderDecoder,LSTMWithEncoderDecoder

import numpy as np

def get_model(model_name, config):
    if model_name == 'RNN' or model_name == 'rnn':
        return RNNWithEncoderDecoder(output_dim=config['output_dim'], embed_dim=config['input_dim'], hidden_dims=config['hidden_dims'],hidden_activations=config['hidden_activations'])
    elif model_name == "lstm" or model_name == "LSTM":
        # dimensions of input, hidden, output must be specified
        return LSTMWithEncoderDecoder(input_dim=config['output_dim'], embed_dim=config['input_dim'], hidden_dims=config['hidden_dims'],hidden_activations=config['hidden_activations'])
    else:
        raise NotImplementedError

# TODO:
def get_activation_data(args):
    trainloader, _, _ = train_models.get_dataloaders(args)
    activation_data = []
    cur_sample_count = 0
    for data, _ in trainloader:
        if cur_sample_count + len(data) < args.activation_batch_size:
            activation_data.append(data)
        else:
            activation_data.append(data[0:(args.activation_batch_size - cur_sample_count)])
            break
        cur_sample_count += len(data)
    return torch.cat(activation_data, dim=0)

class FuseModels:
    def __init__(self, args, train_init): 
        self.args = args
        self.train_init = train_init

        self.base_models = []
        avg_test_acc = 0
        for model_path in args.model_path_list:
            items = model_path.split(',')
            model_name = items[0]
            model_path = items[1]
            state_dict = torch.load(model_path)
            # logging.info("Model: {}, Train Acc: {}, Test Acc: {}".format(model_path,
            #                                                            state_dict['tr_acc'],
            #                                                            state_dict['test_acc']))
            avg_test_acc += state_dict['test_acc']
            cur_model = get_model(model_name, state_dict['config'])
            cur_model.load_state_dict(state_dict['model_state_dict'])
            self.base_models.append(cur_model)
        avg_test_acc /= len(args.model_path_list)
        logging.info("Base models avg acc {}".format(avg_test_acc))

        if args.model_name == 'RNN' or args.model_name == 'rnn' or args.model_name == 'lstm' or args.model_name == 'LSTM': #TODO: check initialization params of RNN model
            config = {'input_dim': args.input_dim, 'hidden_dims': args.hidden_dims,
                      'output_dim': args.output_dim,'hidden_activations':args.hidden_activations}
        else:
            raise NotImplementedError

        self.target_model = get_model(args.model_name, config)
        self.fusion_method = None
        if args.fusion_type == 'avg': # TODO: add avg fusion for RNN
            self.fusion_method = avg_fusion.AvgFusion(args, base_models=self.base_models,
                                                      target_model=self.target_model)
        elif args.fusion_type == 'tlp' and (args.model_name == 'RNN' or args.model_name == 'rnn' or args.model_name == 'lstm' or args.model_name == 'LSTM'):
            TLPFusionClass = tlp_fusion_rnn.TLPFusionRNN
            
            if args.tlp_cost_choice == 'activation': # TODO
                data = get_activation_data(args)
            else:
                data = None
            if args.tlp_init_type == "identity":
                self.target_model = copy.deepcopy(self.base_models[args.tlp_init_model])
            elif args.tlp_init_type == "distill":
                init_fusion = TLPFusionClass(args, base_models=[self.base_models[args.tlp_init_model]],
                                             target_model=self.target_model,
                                             data=data)
                init_fusion.fuse()
                logging.info("Distillation initialization done!")
 
            self.fusion_method = TLPFusionClass(args, base_models=self.base_models,
                                                target_model=self.target_model,
                                                data=data)
        # TODO: add code for tlp without considering hidden states
        elif args.fusion_type == 'tlp_no_hidden' and (args.model_name == 'RNN' or args.model_name == 'rnn' or args.model_name == 'lstm' or args.model_name == 'LSTM'):
            TLPFusionClass = tlp_fusion_rnn.TLPFusionRNNNoHidden
            
            if args.tlp_cost_choice == 'activation': # TODO
                data = get_activation_data(args)
            else:
                data = None
            if args.tlp_init_type == "identity":
                self.target_model = copy.deepcopy(self.base_models[args.tlp_init_model])
            elif args.tlp_init_type == "distill":
                init_fusion = TLPFusionClass(args, base_models=[self.base_models[args.tlp_init_model]],
                                             target_model=self.target_model,
                                             data=data)
                init_fusion.fuse()
                logging.info("Distillation initialization done!")
 
            self.fusion_method = TLPFusionClass(args, base_models=self.base_models,
                                                target_model=self.target_model,
                                                data=data)
        elif args.fusion_type == 'ot' and (args.model_name == 'RNN' or args.model_name == 'rnn' or args.model_name == 'lstm' or args.model_name == 'LSTM'): # TODO: add ot fusion for RNN
            OTFusionClass = ad_hoc_ot_fusion.OTFusion

            if args.ad_hoc_cost_choice == 'activation':
                data = get_activation_data(args)
            else:
                data = None
            if args.ad_hoc_init_type == "distill":
                init_fusion = OTFusionClass(args, base_models=[self.base_models[args.ad_hoc_initialization]],
                                            target_model=self.target_model, data=data)
                init_fusion.fuse()
                logging.info("Distillation initialization done!")
            elif args.ad_hoc_initialization is not None:
                self.target_model = copy.deepcopy(self.base_models[0])
            self.fusion_method = OTFusionClass(args, base_models=self.base_models,
                                               target_model=self.target_model,
                                               data=data)
        elif args.fusion_type == 'fw': # TODO: add fw fusion for RNN
            if args.ad_hoc_cost_choice == 'activation':
                data = get_activation_data(args)
            else:
                data = None
            self.fusion_method = frank_wolfe_fusion.FrankWolfeFusion(args,
                                                                     base_models=self.base_models,
                                                                     target_model=self.target_model,
                                                                     data=data)
        else:
            raise NotImplementedError

    def fuse(self):
        self.fusion_method.fuse()
        self.save_target_model()
        # self.evaluate_target_model() # TODO: immediate evaluation after saving model is disabled for "shakespeare dataset"

    # def evaluate_target_model(self):
    #     evaluate_args = copy.deepcopy(self.args)
    #     evaluate_args.evaluate = True
    #     # TODO: check what's hidden_dims
    #     if self.args.fusion_type == 'fw':
    #         evaluate_args.hidden_dims = self.target_model.get_model_config()['hidden_dims']
    #     evaluate_args.checkpoint_path = os.path.join(self.train_init.model_path, self.args.result_name)

    #     state_dict = torch.load(evaluate_args.checkpoint_path)
    #     model = RNNWithEncoderDecoder(state_dict['config']['output_dim'],state_dict['config']['input_dim'], state_dict['config']['hidden_dims'],state_dict['config']['hidden_activations'])
    #     model.load_state_dict(state_dict['model_state_dict'])
    #     model.cuda()

    #     print("target model channels:",model.channels)

    #     dataset = RNNDataset(device='cuda' if evaluate_args.cuda else 'cpu', \
    #                 batch_size=evaluate_args.eval_batch_size, \
    #                 eval_batch_size=evaluate_args.eval_batch_size, \
    #                 dataset_name= evaluate_args.dataset_name,train_data_path='/train.csv')
    #     trainer = Trainer(model=model,dataset=dataset,loss_fn="cross_entropy",accuracy_fn="acc_classification", \
    #         batch_size=evaluate_args.eval_batch_size,eval_batch_size=evaluate_args.eval_batch_size,device='cuda' if evaluate_args.cuda else 'cpu')
        
    #     print("tlp_sinkhorn_regularization",evaluate_args.tlp_sinkhorn_regularization)
    #     trainer.train_epoch(train=False)
    #     for k, v in trainer.diagnostics.items():
    #         print(f'| {k:25} | {v:25} |')
        
    #     logging.info('Evaluation done.')

    def save_target_model(self):
        save_path = os.path.join(self.train_init.model_path, self.args.result_name)
        print("save_path:",save_path)
        torch.save({'model_state_dict': self.target_model.state_dict(),
                    'config': self.target_model.get_model_config()},
                   save_path)
        torch.save({'model_state_dict': self.target_model.state_dict(),
                    'config': self.target_model.get_model_config()},
                   "./fused_models/"+self.args.result_name)
        logging.info('Model saved at {}'.format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='RNN')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--result_name', type=str, default='')

    # parser.add_argument('--data_path', type=str, default='./data')
    # parser.add_argument('--optimizer', type=str, default='Adam')
    # parser.add_argument("--lr", default=1e-3, type=float)
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--lr_scheduler', type=str, default='StepLR',
    #                     choices=['StepLR', 'MultiStepLR'])
    # parser.add_argument('--lr_step_size', type=int, default=10000)
    # parser.add_argument('--lr_gamma', type=float, default=1.0)
    # parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    # parser.add_argument('--momentum', type=float, default=0)

    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--hidden_dims', type=str, default=None)
    parser.add_argument('--output_dim', type=int, default=None)
    parser.add_argument('--hidden_activations', type=str, default=None)

    # parser.add_argument('--log_step', type=int, default=100,
    #                     help='The steps after which models would be logged.')

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
    hid_dims = np.arange(128,384+1,16)
    base_result_name = args.result_name
    for hid_dim in hid_dims:
        print("Current hid_dim is:",hid_dim)

        args.hidden_dims = [hid_dim]
        args.result_name = base_result_name + "_hid" + str(hid_dim) + ".pth"

        train_init = init.Init(args=args, run_str=run_str)
        fuse_models = FuseModels(args, train_init)
        fuse_models.fuse()
        
if __name__ == "__main__":
    print("torch.cuda.is_available()",torch.cuda.is_available())
    main()
