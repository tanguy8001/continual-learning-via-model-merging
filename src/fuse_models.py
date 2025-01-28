import argparse
import copy
import logging
import os
import torch

import ad_hoc_ot_fusion
import avg_fusion
import init
import model
import train_models

import curve_merging
from models import mlpnet, fcmodel

def get_model(model_name, config):
    if model_name == 'FC':
        return fcmodel.FCModelBase(config['input_dim'], config['hidden_dims'], config['output_dim'])
    elif model_name == 'MlpNet':
        return mlpnet.MlpNetBase(input_dim=config['input_dim'], num_classes=config['output_dim'])
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
        else:
            raise NotImplementedError


        self.target_model = get_model(args.model_name, config)
        self.fusion_method = None

        if args.fusion_type == 'avg':
            print("Performing vanilla merging!")
            self.fusion_method = avg_fusion.AvgFusion(args, base_models=self.base_models,
                                                      target_model=self.target_model)

        elif args.fusion_type == 'ot':
            print(f"Performing OT merging, using the {args.ad_hoc_cost_choice} cost choice!")
            OTFusionClass = ad_hoc_ot_fusion.OTFusion
            if args.ad_hoc_cost_choice == 'activation':
                data = get_activation_data(args)
            else:
                data = None
            if args.ad_hoc_initialization is not None:
                self.target_model = copy.deepcopy(self.base_models[0])
            self.fusion_method = OTFusionClass(args, base_models=self.base_models,
                                               target_model=self.target_model,
                                               data=data)

        elif args.fusion_type == 'curve':
            print("Performing curve merging!")
            CURVEFusionClass = curve_merging.CurveFusion
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
        val_acc, test_acc = trainer.evaluate()
        
        #model_path = "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints"
        #final_save_path = os.path.join(model_path, 'final_curve_fusion_model.pth')
        #config = curve_merging.CurveConfig()
        #self.save_model(self.target_model, config, config.epochs, val_acc, test_acc, final_save_path)

        logging.info('Evaluation done.')

    def save_target_model(self):
        save_path = os.path.join(self.train_init.model_path, 'fused_model.pth')
        torch.save({'model_state_dict': self.target_model.state_dict(),
                    'config': self.target_model.get_model_config()},
                   save_path)
        logging.info('Model saved at {}'.format(save_path))

    def save_model(self, model, config, epoch, val_acc, test_acc, save_path):
        torch.save({
            'epoch': epoch,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'model_state_dict': model.state_dict(),
            'config': model.get_model_config()
        }, save_path)


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

    parser.add_argument('--ad_hoc_cost_choice', type=str, default='activation',
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
    
    if args.fusion_type == 'ot':
        run_params.extend(['cost_choice', args.ad_hoc_cost_choice,
                           'solver', args.ad_hoc_ot_solver])
        if args.use_pre_activations:
            run_params.extend(['preact'])
        if args.ad_hoc_initialization is not None:
            run_params.extend(['init', args.ad_hoc_initialization])
        if args.ad_hoc_ot_solver == 'sinkhorn':
            run_params.extend(['reg', args.ad_hoc_sinkhorn_regularization])

    run_str = '_'.join([str(x) for x in run_params])
    
    train_init = init.Init(args=args, run_str=run_str)
    fuse_models = FuseModels(args, train_init)
    fuse_models.fuse()

if __name__ == "__main__":
    main()

