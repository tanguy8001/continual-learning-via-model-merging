import copy
import logging
import pdb
import torch
import argparse

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from utils import ot_algorithms
from utils import memory_safe_methods
#from tlp_rnn_fusion.rnn_models import RNNModel


class GWFusionRNN:
    """
    This class currently assumes that the base_models and the target_model
    has the same model type and differ in the number of nodes per layer

    base_models: List of base models
    target_model: Target model
    """

    def __init__(self, args, base_models, target_model, data):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model
        self.data = data

    def fuse(self):
        logging.info('Starting RNN model fusion.')
        # Fuses all the layers into the target model.
        if torch.cuda.is_available():
            for model in self.base_models:
                model.cuda()
            self.target_model.cuda()
            if self.data is not None:
                self.data = self.data.cuda()

        prev_pi = []
        input_dim = self.target_model.input_dim
        for i in range(len(self.base_models)):
            # For the input layer identity is the coupling between the nodes.
            pi = torch.eye(input_dim, dtype=torch.float) / (1.0 * input_dim)
            if torch.cuda.is_available():
                pi = pi.cuda()
            prev_pi.append(pi)

        for i in range(1, self.target_model.num_layers + 1):
            cur_pi = self.fuse_single_layer(layer=i, prev_pi=prev_pi)
            if self.args.tlp_cost_choice == 'activation':
                prev_pi = self.get_activation_coupling(layer=i)
            else:
                prev_pi = cur_pi  # Other logic follows here
        logging.info('Model fusion for RNN completed.')

    def fuse_single_layer(self, layer, prev_pi):
        logging.info('Fusing layer {}'.format(layer))
        # Fuses a singe layer of the networks.
        base_Ws = []
        base_Hs = []
        for model in self.base_models:
            Ws, Hs = model.get_layer_weights(layer_num=layer)
            base_Ws.append(Ws)
            base_Hs.append(Hs)
        target_Ws, target_Hs = self.target_model.get_layer_weights(layer_num=layer) #TODO: add a new method to get W and H
        if layer == self.target_model.num_layers: # output linear layer
            return self.fuse_single_linear_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_Ws, target_Ws,prev_pi)
        else: # hidden layer
            return self.fuse_single_hidden_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_Ws,base_Hs, target_Ws,target_Hs,prev_pi)

    def fuse_single_hidden_layer_helper(self, num_models, layer, is_last_layer, base_Ws,base_Hs, target_Ws,target_Hs, prev_pi):
        # Fuses a singe layer of the networks.
        logging.info('Helper fusing layer {}'.format(layer))

        Ws = copy.deepcopy(target_Ws.data)
        Hs = copy.deepcopy(target_Hs.data)
        if torch.cuda.is_available():
            Ws = Ws.cuda()
            Hs = Hs.cuda()

        if is_last_layer:
            # For the last layer, we know the coupling for the final layer.
            pi = []
            for i in range(num_models):
                n = target_Ws.size(0)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available():
                    cur_pi = cur_pi.cuda()
                pi.append(cur_pi)
            Ws, Hs = self.solve_hidden_weights(base_Ws,base_Hs,prev_pi, pi)
            target_Ws.data = Ws.data
            target_Hs.data = Hs.data
            return pi

        # When this is not the last layer, we iteratively optimize
        # for pi and weights.
        pi = []
        for W_i in base_Ws:
            n = target_Ws.size(0) # dimension of current layer in the target model, should be equal to target_Hs.size(0)
            m = W_i.size(0) # dimension of current layer in base model i, should be equal to  H_i.size(0))
            cur_pi = torch.ones(size=(n, m), dtype=torch.float) / (1.0 * n * m)
            if torch.cuda.is_available():
                cur_pi = cur_pi.cuda()
            pi.append(cur_pi)

        max_iter = 100
        threshold = 1e-5
        actual_iter = max_iter
        for i in range(1, max_iter + 1):
            new_pi = self.solve_pi(base_Ws, base_Hs, target_Ws, target_Hs, prev_pi, cur_pi, layer == 1)
            new_Ws, new_Hs = self.solve_hidden_weights(base_Ws, base_Hs, prev_pi, new_pi)

            Ws_epsilon = (Ws - new_Ws).pow(2).mean()
            Hs_epsilon = (Hs - new_Hs).pow(2).mean()
            pi_epsilon = 0
            for new_pi_i, pi_i in zip(new_pi, pi):
                pi_epsilon += (new_pi_i - pi_i).pow(2).mean()
            Ws = new_Ws
            Hs = new_Hs
            pi = new_pi
            if Ws_epsilon < threshold and Hs_epsilon < threshold and pi_epsilon < threshold:
                logging.info('Ws_epsilon {}, Hs_epsilon {}, pi_epsilon {}'.format(Ws_epsilon, Hs_epsilon, pi_epsilon))
                actual_iter = i
                break
        target_Ws.data = Ws.data
        target_Hs.data = Hs.data
        logging.info('Num of actual iterations={}'.format(actual_iter))
        return pi

    def fuse_single_linear_layer_helper(self, num_models, layer, is_last_layer, base_weights,
                                 target_weights, prev_pi):
        logging.info('Helper fusing layer {}'.format(layer))
        # Fuses a singe layer of the networks.
        weights = copy.deepcopy(target_weights.data)
        # weights = torch.rand_like(target_weights)
        if torch.cuda.is_available():
            weights = weights.cuda()

        if is_last_layer:
            # For the last layer, we know the coupling for the final layer.
            pi = []
            for i in range(num_models):
                n = target_weights.size(0)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available():
                    cur_pi = cur_pi.cuda()
                pi.append(cur_pi)
            weights = self.solve_linear_weights(base_weights, prev_pi, pi)
            target_weights.data = weights.data
            return pi

        # When this is not the last layer, we iteratively optimize
        # for pi and weights.
        pi = []
        for cur_weight in base_weights:
            n = target_weights.size(0)
            m = cur_weight.size(0)
            cur_pi = torch.ones(size=(n, m), dtype=torch.float) / (1.0 * n * m)
            if torch.cuda.is_available():
                cur_pi = cur_pi.cuda()
            pi.append(cur_pi)

        max_iter = 100
        threshold = 1e-5
        actual_iter = max_iter
        for i in range(1, max_iter + 1):
            new_pi = self.solve_pi(base_weights, weights, prev_pi, layer == 1)
            new_weights = self.solve_linear_weights(base_weights, prev_pi, new_pi)

            weights_epsilon = (weights - new_weights).pow(2).mean()
            pi_epsilon = 0
            for new_pi_i, pi_i in zip(new_pi, pi):
                pi_epsilon += (new_pi_i - pi_i).pow(2).mean()
            weights = new_weights
            pi = new_pi
            if weights_epsilon < threshold and pi_epsilon < threshold:
                logging.info('weights_epsilon {}, pi_epsilon {}'.format(weights_epsilon, pi_epsilon))
                actual_iter = i
                break
        target_weights.data = weights.data
        logging.info('Num of actual iterations={}'.format(actual_iter))
        return pi

    def solve_pi_helper(self, base_Ws, base_Hs, target_Ws, target_Hs, prev_pi, cur_pi, is_first_layer=False):
        pi = []
        with torch.no_grad():
            if is_first_layer:
                Ws = target_Ws.unsqueeze(1) # dim = (cur_dim_W,1,pre_dim_W)
                Hs = target_Hs.unsqueeze(1).unsqueeze(3) # (cur_dim_H,1,cur_dim_H) -> (cur_dim_H,1,cur_dim_H,1)
                for W_i, H_i,cur_pi_i in zip(base_Ws, base_Hs, cur_pi):
                    W_i = W_i.unsqueeze(0) # dim = (1,cur_dim_Wi,pre_dim_Wi) = (1,cur_dim_Wi,pre_dim_W)
                    H_i = H_i.unsqueeze(0).unsqueeze(2) # (1, cur_dim_Hi, cur_dim_Hi) -> (1, cur_dim_Hi, 1, cur_dim_Hi)
                    diff_W = (Ws - W_i).pow(2) # dim = (cur_dim_W,cur_dim_Wi,pre_dim_W)
                    diff_H = (Hs - H_i).pow(2) # dim = (cur_dim_H, cur_dim_Hi, cur_dim_H, cur_dim_Hi)
                    cost = diff_W.sum(-1) + (diff_H * cur_pi_i[0]).sum(-1).sum(-1)
                
                    # Different algorithms for solving Linear equations goes here.
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi
            else:
                Ws = target_Ws.unsqueeze(1).unsqueeze(3) 
                Hs = target_Hs.unsqueeze(1).unsqueeze(3)

                use_loop = False
                for W_i, H_i, prev_pi_i, cur_pi_i in zip(base_Ws, base_Hs, prev_pi, cur_pi):
                    W_i = W_i.unsqueeze(0).unsqueeze(2) 
                    H_i = H_i.unsqueeze(0).unsqueeze(2)
                    if use_loop:
                        cost_W = memory_safe_methods.get_cost(Ws, W_i, prev_pi_i) 
                        cost_H = memory_safe_methods.get_cost(Hs, H_i, cur_pi_i) 
                        cost = cost_W + cost_H
                    else:
                        try:
                            diff_W = (Ws - W_i).pow(2)
                            diff_H = (Hs - H_i).pow(2)
                            cost = (diff_W * prev_pi_i[0]).sum(-1).sum(-1) + (diff_H * cur_pi_i[0]).sum(-1).sum(-1)
                        except RuntimeError as e:
                            error = "{}".format(e)
                            if error.startswith("CUDA out of memory."):
                                cost_W = memory_safe_methods.get_cost(Ws, W_i, prev_pi_i) 
                                cost_H = memory_safe_methods.get_cost(Hs, H_i, cur_pi_i) 
                                cost = cost_W + cost_H
                                use_loop = True
                            else:
                                print(error)
                                raise ImportError(e)
                    # Different algorithms for solving Linear equations goes here.
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi

    def solve_pi(self, base_Ws, base_Hs, target_Ws, target_Hs, prev_pi, cur_pi, is_first_layer=False):
        max_iter = 100
        actual_iter = max_iter
        threshold = 1e-5
        for i in range(1, max_iter + 1):
            new_pi = self.solve_pi_helper(base_Ws, base_Hs, target_Ws, target_Hs, prev_pi, cur_pi, is_first_layer)
            pi_epsilon = 0
            for new_pi_i, cur_pi_i in zip(new_pi, cur_pi):
                pi_epsilon += (new_pi_i - cur_pi_i).pow(2).mean()
            cur_pi = new_pi
            if pi_epsilon < threshold:
                logging.info('pi_epsilon for solving pi={}'.format(pi_epsilon))
                actual_iter = i
                break
        logging.info('Num of actual iterations for solving pi={}'.format(actual_iter))
        return cur_pi
    

    def solve_hidden_weights(self, base_Ws,base_Hs,prev_pi, pi):
        k_l = pi[0].size(0)
        k_l_prev = prev_pi[0].size(0)
        Hs = None
        Ws = None
        if 'model_weights' not in self.args.__dict__ or self.args.model_weights is None:
            model_weights = [1.0 / len(pi)] * len(pi)
        else:
            model_weights = self.args.model_weights

        for i, (pi_i, prev_pi_i, w_i, h_i) in enumerate(zip(pi, prev_pi, base_Ws, base_Hs)):  
            # calculate matrix W
            Ws_i = torch.matmul(pi_i, torch.matmul(w_i, k_l * k_l_prev * prev_pi_i.transpose(0, 1)))
            Ws_i *= model_weights[i]
            if Ws is None:
                Ws = Ws_i
            else:
                Ws += Ws_i
            # calculate matrix H
            Hs_i = torch.matmul(pi_i, torch.matmul(h_i, k_l * k_l * pi_i.transpose(0, 1)))
            Hs_i *= model_weights[i]
            if Hs is None:
                Hs = Hs_i
            else:
                Hs += Hs_i
        return Ws, Hs

    def solve_linear_weights(self, base_weights, prev_pi, pi):
        k_l = pi[0].size(0)
        k_l_prev = prev_pi[0].size(0)
        weights = None
        if 'model_weights' not in self.args.__dict__ or self.args.model_weights is None:
            model_weights = [1.0 / len(pi)] * len(pi)
        else:
            model_weights = self.args.model_weights

        for i, (pi_i, prev_pi_i, w_i) in enumerate(zip(pi, prev_pi, base_weights)):
            weights_i = torch.matmul(pi_i, torch.matmul(w_i, k_l * k_l_prev * prev_pi_i.transpose(0, 1)))
            weights_i *= model_weights[i]
            if weights is None:
                weights = weights_i
            else:
                weights += weights_i
        return weights

    def get_activation_coupling(self, layer):
        with torch.no_grad():
            target_model_activations = self.target_model.get_activations(self.data,
                                                                         layer_num=layer)
            target_model_activations = target_model_activations.reshape(target_model_activations.size(0), -1)
            pi = []
            for model in self.base_models:
                activations = model.get_activations(self.data, layer_num=layer,
                                                    pre_activations=self.args.use_pre_activations)
                activations = activations.reshape(activations.size(0), -1)
                # cost = (target_model_activations.unsqueeze(1) - activations.unsqueeze(0)).pow(2).sum(-1)
                cost = memory_safe_methods.get_activation_cost(target_model_activations, activations)
                # Different algorithms for solving Linear equations can go here.
                cur_pi = self.ot_solver(cost)
                pi.append(cur_pi)
        return pi

    def ot_solver(self, cost):
        if self.args.tlp_ot_solver == 'sinkhorn':
            epsilon = self.args.tlp_sinkhorn_regularization
            pi, _ = ot_algorithms.sinkhorn_coupling(cost, epsilon=epsilon, niter=100)
            return pi
        elif self.args.tlp_ot_solver == 'emd':
            pi, _ = ot_algorithms.ot_emd_coupling(cost)
            return pi
        else:
            raise NotImplementedError


"""
for testing purpose
"""
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--experiment_name', type=str, default='test')
#     parser.add_argument('--model_name', type=str, default='FC')
#     parser.add_argument('--dataset_name', type=str, default='MNIST')
#     parser.add_argument('--result_path', type=str, default='result')

#     parser.add_argument('--data_path', type=str, default='./data')
#     parser.add_argument('--optimizer', type=str, default='Adam')
#     parser.add_argument("--lr", default=1e-3, type=float)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
#     parser.add_argument("--num_epochs", default=300, type=int)
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--lr_scheduler', type=str, default='StepLR',
#                         choices=['StepLR', 'MultiStepLR'])
#     parser.add_argument('--lr_step_size', type=int, default=10000)
#     parser.add_argument('--lr_gamma', type=float, default=1.0)
#     parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
#     parser.add_argument('--momentum', type=float, default=0)

#     parser.add_argument('--input_dim', type=int, default=784)
#     parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
#     parser.add_argument('--output_dim', type=int, default=10)

#     parser.add_argument('--log_step', type=int, default=100,
#                         help='The steps after which models would be logged.')

#     parser.add_argument('--evaluate', default=False, action='store_true')
#     parser.add_argument('--resume', default=False, action='store_true')
#     parser.add_argument('--checkpoint_path', type=str, default=None)

#     parser.add_argument('--no_cuda', default=False, action='store_true')
#     parser.add_argument('--gpu_ids', type=str, default='0')
#     parser.add_argument("--seed", default=24601, type=int)

#     parser.add_argument('--model_path_list', type=str, default=None, nargs='+',
#                         help="Comma separated list of models and checkpoints"
#                              "to be used fused together")

#     # Fusion parameters
#     parser.add_argument('--fusion_type', type=str, default=None,
#                         choices=['tlp', 'avg', 'ot', 'fw'])
#     parser.add_argument('--activation_batch_size', type=int, default=100)
#     parser.add_argument('--use_pre_activations', default=False, action='store_true')
#     parser.add_argument('--model_weights', default=None, type=float, nargs='+',
#                         help='Comma separated list of weights for each model in fusion')

#     parser.add_argument('--tlp_cost_choice', type=str, default='weight',
#                         choices=['weight', 'activation'])
#     parser.add_argument('--tlp_ot_solver', type=str, default='sinkhorn',
#                         choices=['sinkhorn', 'emd'])
#     parser.add_argument('--tlp_sinkhorn_regularization', type=float, default=0.001)
#     parser.add_argument('--tlp_init_type', type=str, default=None,
#                         choices=[None, 'identity', 'distill'])
#     parser.add_argument('--tlp_init_model', type=int, default=None)

#     parser.add_argument('--ad_hoc_cost_choice', type=str, default='weight',
#                         choices=['weight', 'activation'])
#     parser.add_argument('--ad_hoc_ot_solver', type=str, default='sinkhorn',
#                         choices=['sinkhorn', 'emd'])
#     parser.add_argument('--ad_hoc_sinkhorn_regularization', type=float, default=0.001)
#     parser.add_argument('--ad_hoc_init_type', type=str, default=None,
#                         choices=[None, 'distill'])
#     parser.add_argument('--ad_hoc_initialization', type=int, default=None)

#     parser.add_argument('--fw_cost_choice', type=str, default=None)
#     parser.add_argument('--fw_sinkhorn_regularization', type=float, default=0.01)
#     parser.add_argument('--fw_single_layer_fusion_type', type=str, default=None,
#                         choices=[None, 'support'])
#     parser.add_argument('--fw_minimization_type', type=str, default=None,
#                         choices=[None, 'reg', 'pgd'],
#                         help="The type of minimization - None, with regularization or PGD")

#     parser.add_argument('--resnet_skip_connection_handling', type=str, default='pre',
#                         choices=['pre', 'post'],
#                         help='Pre means use pis from previously similar layer, post means handle later')
#     parser.add_argument('--resnet_use_max_pool', default=False, action='store_true')

#     parser.add_argument('--vgg_cfg', type=str, default='A', choices=['S', 'A'])
#     parser.add_argument('--resnet_cfg', type=str, default='R', choices=['S', 'R'])

#     args = parser.parse_args()

#     base_model_1 = RNNModel(2,[2,7,8],4,None)
#     base_model_2 = RNNModel(2,[3,4,6],4,None)
#     base_models = [base_model_1,base_model_2]
#     target_model = RNNModel(2,[4,7,9],4,None)

#     fusion_method = TLPFusionRNN(args,base_models=base_models,target_model=target_model,data=None)
#     fusion_method.fuse()

                        
