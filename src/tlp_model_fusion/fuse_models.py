import argparse
import copy
import logging
import os
import torch

import ad_hoc_ot_fusion
import avg_fusion
import init
import model
import resnet_models
import tlp_fusion
import train_models
import vgg_models
import gw_fusion_rnn

import curve_merging
from models import mlpnet

def get_model(model_name, config):
    if model_name == 'FC':
        #return model.FCModel(config['input_dim'], config['hidden_dims'], config['output_dim'])
        return mlpnet.MlpNetBase(input_dim=config['input_dim'], num_classes=config['output_dim'])
    elif model_name == 'Conv':
        return model.ConvModel(input_channels=config['input_dim'], output_dim=config['output_dim'])
    elif model_name == 'vgg11':
        return vgg_models.vgg11(num_classes=config['num_classes'])
    elif model_name == 'resnet18':
        return resnet_models.resnet18(num_classes=config['num_classes'],
                                      use_max_pool=config['use_max_pool'])
    elif model_name == 'ImageRNN':
        return model.ImageRNN(n_steps=config['n_steps'], n_inputs=config['n_inputs'],
                              n_neurons=config['n_neurons'], n_outputs=config['n_outputs'],
                              act_type=config['act_type'], step_start=config['step_start'])
    else:
        raise NotImplementedError


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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.base_models = []
        avg_test_acc = 0
        print(args.model_path_list)
        for model_path in args.model_path_list:
            items = model_path.split(',')
            model_name = items[0]
            model_path = items[1]
            state_dict = torch.load(model_path, map_location=device)
            logging.info("Model: {}, Val Acc: {}, Test Acc: {}".format(model_path,
                                                                       state_dict['val_acc'],
                                                                       state_dict['test_acc']))
            avg_test_acc += state_dict['test_acc']
            cur_model = get_model(model_name, state_dict['config'])
            cur_model.load_state_dict(state_dict['model_state_dict'])
            self.base_models.append(cur_model)
        avg_test_acc /= len(args.model_path_list)
        logging.info("Base models avg acc {}".format(avg_test_acc))

        if args.model_name == 'FC':
            config = {'input_dim': args.input_dim, 'hidden_dims': args.hidden_dims,
                      'output_dim': args.output_dim}
        elif args.model_name == 'Conv':
            config = {'input_dim': 1 if args.dataset_name == 'MNIST' else 3,
                      'output_dim': args.output_dim}
        elif args.model_name == 'vgg11':
            config = {'num_classes': 10}
        elif args.model_name == 'resnet18':
            config = {'num_classes': 10, 'use_max_pool': args.resnet_use_max_pool}
        elif args.model_name == 'ImageRNN':
            config = {'n_steps': args.rnn_steps, 'n_inputs': args.input_dim,
                      'n_outputs': args.output_dim, 'n_neurons': args.hidden_dims,
                      'act_type': args.rnn_act_type, 'step_start': args.rnn_step_start}
        else:
            raise NotImplementedError

        self.target_model = get_model(args.model_name, config)
        self.fusion_method = None
        if args.fusion_type == 'avg':
            print("Performing vanilla merging!")
            self.fusion_method = avg_fusion.AvgFusion(args, base_models=self.base_models,
                                                      target_model=self.target_model)
        elif args.fusion_type == 'tlp':
            TLPFusionClass = tlp_fusion.TLPFusion
            if args.model_name == 'vgg11':
                TLPFusionClass = tlp_fusion.TLPFusionVGG
            elif args.model_name == 'resnet18':
                TLPFusionClass = tlp_fusion.TLPFusionResnet
            elif args.model_name == 'ImageRNN':
                TLPFusionClass = tlp_fusion.TLPFusionRNN

            if args.tlp_cost_choice == 'activation':
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
        elif args.fusion_type == 'ot':
            print("Performing OT merging!")
            OTFusionClass = ad_hoc_ot_fusion.OTFusion
            if args.model_name == 'vgg11':
                OTFusionClass = ad_hoc_ot_fusion.OTFusionVGG
            elif args.model_name == 'resnet18':
                OTFusionClass = ad_hoc_ot_fusion.OTFusionResnet
            elif args.model_name == 'ImageRNN':
                OTFusionClass = ad_hoc_ot_fusion.OTFusionRNN

            if args.ad_hoc_cost_choice == 'activation':
                data = get_activation_data(args)
            else:
                data = None
            if args.ad_hoc_initialization is not None:
                self.target_model = copy.deepcopy(self.base_models[0])
            self.fusion_method = OTFusionClass(args, base_models=self.base_models,
                                               target_model=self.target_model,
                                               data=data)
        elif args.fusion_type == 'gw':
            GWFusionClass = gw_fusion_rnn.GWFusionRNN

            if args.tlp_cost_choice == 'activation':
                data = get_activation_data(args)
            else:
                data = None
            if args.tlp_init_type == "identity":
                self.target_model = copy.deepcopy(self.base_models[args.tlp_init_model])
            elif args.tlp_init_type == "distill":
                init_fusion = GWFusionClass(args, base_models=[self.base_models[args.tlp_init_model]],
                                             target_model=self.target_model,
                                             data=data)
                init_fusion.fuse()
                logging.info("Distillation initialization done!")

            self.fusion_method = GWFusionClass(args, base_models=self.base_models,
                                                target_model=self.target_model,
                                                data=data)
        elif args.fusion_type == 'curve':
            print("Performing curve merging!")

            CURVEFusionClass = curve_merging.CurveFusion

            self.target_model = copy.deepcopy(self.base_models[0])

            trainloader, valloader, testloader = train_models.get_dataloaders(args)
            data = {
                'train': trainloader,
                'val': valloader,
                'test': testloader,
            }

            self.fusion_method = CURVEFusionClass(args, base_models=self.base_models,
                                                target_model=self.target_model,
                                                data=data)
            
        else:
            raise NotImplementedError

    def fuse(self):
        self.fusion_method.fuse()
        self.save_target_model()
        self.evaluate_target_model()

    def evaluate_target_model(self):
        evaluate_args = copy.deepcopy(self.args)
        evaluate_args.evaluate = True
        if self.args.fusion_type == 'fw':
            evaluate_args.hidden_dims = self.target_model.get_model_config()['hidden_dims']
        evaluate_args.checkpoint_path = os.path.join(self.train_init.model_path, 'fused_model.pth')
        trainer = train_models.Trainer(self.train_init, evaluate_args)
        trainer.evaluate()
        logging.info('Evaluation done.')

    def save_target_model(self):
        save_path = os.path.join(self.train_init.model_path, 'fused_model.pth')
        torch.save({'model_state_dict': self.target_model.state_dict(),
                    'config': self.target_model.get_model_config()},
                   save_path)
        logging.info('Model saved at {}'.format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='FC')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--result_path', type=str, default='result')

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'MultiStepLR'])
    parser.add_argument('--lr_step_size', type=int, default=10000)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    parser.add_argument('--momentum', type=float, default=0)

    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--rnn_steps', type=int, default=1,
                        help='Number of steps that RNN executes')
    parser.add_argument('--rnn_act_type', type=str, default='tanh',
                        choices=['tanh', 'relu'])
    parser.add_argument('--rnn_step_start', type=int, default=0,
                        help='Step number to start with for RNN experiments, helper flag')

    parser.add_argument('--log_step', type=int, default=100,
                        help='The steps after which models would be logged.')

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
                        choices=['tlp', 'avg', 'ot', 'fw', 'gw', 'curve'])
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
    parser.add_argument('--ad_hoc_init_type', type=str, default=None)
    parser.add_argument('--ad_hoc_initialization', type=int, default=None)

    parser.add_argument('--resnet_skip_connection_handling', type=str, default='pre',
                        choices=['pre', 'post'],
                        help='Pre means use pis from previously similar layer, post means handle later')
    parser.add_argument('--resnet_use_max_pool', default=False, action='store_true')

    parser.add_argument('--theta_pi', type=float, default=1.0)
    parser.add_argument('--theta_w', type=float, default=1.0)
    parser.add_argument('--auto_optimize', type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(level=logging.INFO)

    run_params = ['fusion', args.fusion_type,
                  'num_models', len(args.model_path_list),
                  'layers', len(args.hidden_dims),
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

    elif args.fusion_type == 'gw':
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
    if args.model_name == 'resnet18':
        run_params.extend(['skip_conn', args.resnet_skip_connection_handling])

    run_str = '_'.join([str(x) for x in run_params])
    
    train_init = init.Init(args=args, run_str=run_str)
    fuse_models = FuseModels(args, train_init)
    fuse_models.fuse()

if __name__ == "__main__":
    main()

