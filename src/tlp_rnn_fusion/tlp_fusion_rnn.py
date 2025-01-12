import copy
import logging
import pdb
import torch
import argparse
import os

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from src.tlp_model_fusion.utils import ot_algorithms
from src.tlp_model_fusion.utils import memory_safe_methods
from src.tlp_rnn_fusion import rnn_models
from src.tlp_rnn_fusion import fuse_rnn_models

def for_loop_cost(w, w_i, prev_pi_i,LSTM=False):
    """
        w: 
        w_i:
        prev_pi_i:
    """
    cur_dim = w.size(-4)
    prev_dim = w.size(-2)
    cur_dim_i = w_i.size(-3)
    prev_dim_i = w_i.size(-1)

    if LSTM:
        diff = torch.zeros((4, cur_dim,cur_dim_i,prev_dim,prev_dim_i))
        for index_0 in range(4):
            for index_1 in range(cur_dim):
                for index_2 in range(cur_dim_i):
                    for index_3 in range(prev_dim):
                        for index_4 in range(prev_dim_i): 
                            diff[index_0,index_1,index_2,index_3,index_4] = (w[index_0,index_1,0,index_3,0] - w_i[index_0,0,index_2,0,index_4]) ** 2
        cost = torch.zeros((4,cur_dim,cur_dim_i))
        for index_0 in range(4):
            for index_1 in range(cur_dim):
                for index_2 in range(cur_dim_i):
                    cost[index_0,index_1,index_2] += torch.sum(diff[index_0,index_1,index_2] * prev_pi_i)
    else:
        diff = torch.zeros((cur_dim,cur_dim_i,prev_dim,prev_dim_i))
        for index_0 in range(cur_dim):
            for index_1 in range(cur_dim_i):
                for index_2 in range(prev_dim):
                    for index_3 in range(prev_dim_i): 
                        diff[index_0,index_1,index_2,index_3] = (w[index_0,0,index_2,0] - w_i[0,index_1,0,index_3]) ** 2   
        cost = torch.zeros((cur_dim,cur_dim_i))
        for index_0 in range(cur_dim):
            for index_1 in range(cur_dim_i):
                cost[index_0,index_1] += torch.sum(diff[index_0,index_1] * prev_pi_i)
    
    return cost

class TLPFusionRNN:
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

        # alpha_h : The weight for cost of the hidden layer
        if 'alpha_h' in args.__dict__:
            self.alpha_h_arr = args.alpha_h
        else:
            self.alpha_h_arr = [1.0] * len(self.target_model.num_layers)
        self.alpha_h = 1.0
        # logging.info("Hidden cost weight:{}".format(','.join(self.alpha_h_arr)))

        # niters_rnn: number of inner loop optimization
        if 'niters_rnn' in args.__dict__:
            self.niters_rnn = args.niters_rnn
        else:
            self.niters_rnn = 100
        logging.info("Hidden niters:{}".format(self.niters_rnn))

        # reg_gamma : Multiplicative factor for reg
        if 'reg_gamma' in args.__dict__:
            self.reg_gamma = args.reg_gamma
        else:
            self.reg_gamma = 1.0
        logging.info("Reg gamma:{}".format(self.reg_gamma))

        if 'tlp_sinkhorn_regularization_list' not in args.__dict__ \
            or self.args.tlp_sinkhorn_regularization_list is None:
            self.tlp_sinkhorn_reg_list = [self.args.tlp_sinkhorn_regularization] * len(self.target_model.num_layers)
        else:
            self.tlp_sinkhorn_reg_list = self.args.tlp_sinkhorn_regularization_list
        self.tlp_sinkhorn_reg = 0.01

        if "min_reg" in args.__dict__:
            self.min_reg = self.args.min_reg
        else:
            self.min_reg = 0.001

        self.theta_pi = 1.0
        if "theta_pi" in args.__dict__:
            self.theta_pi = self.args.theta_pi

        self.num_pi_iters = 100
        if "num_pi_iters" in args.__dict__:
            self.num_pi_iters = self.args.num_pi_iters

    def fuse(self):
        logging.info('Starting RNN model fusion.')
        # Fuses all the layers into the target model.
        if torch.cuda.is_available() and self.args.device == 'cuda':
            for model in self.base_models:
                model.cuda()
            self.target_model.cuda()
            if self.data is not None:
                self.data = self.data.cuda()

        # initialization - the first layer
        prev_pis = []
        pis_model_1 = [] # couplings for model 1
        pis_model_2 = [] # couplings for model 2

        if self.args.encoder:
            initial_dimension = self.target_model.input_dim
        else:
            initial_dimension = self.target_model.channels[0]
        # self.target_model.get_layer_weights(1).size(1)
        for i in range(len(self.base_models)):
            # For the input layer identity is the coupling between the nodes.
            pi = torch.eye(initial_dimension, dtype=torch.float) / (1.0 * initial_dimension)
            if torch.cuda.is_available() and self.args.device == 'cuda':
                pi = pi.cuda()
            prev_pis.append(pi)
        pis_model_1.append(pi)
        pis_model_2.append(pi)

        for i in range(1, self.target_model.num_layers + 1):
            self.tlp_sinkhorn_reg = self.args.tlp_sinkhorn_regularization  # reset each layer
            cur_pi = self.fuse_single_layer(layer=i, prev_pis=prev_pis)
            pis_model_1.append(cur_pi[0]) # get couplings for model 1
            pis_model_2.append(cur_pi[1]) # get couplings for model 2
            if self.args.tlp_cost_choice == 'activation':
                prev_pis = self.get_activation_coupling(layer=i)
            else:
                prev_pis = cur_pi  # Other logic follows here

        for i in range(1, self.target_model.num_layers + 1):
            self.compute_weights_diff(layer=i)

        logging.info('Begin to generate permuted model 1.')
        permuted_model_1_path = os.path.join(self.args.save_path, 'permuted_model_1.pth')
        self.generate_permuted_model(self.base_models[0], pis_model_1, permuted_model_1_path)
        logging.info('Finish generating permuted model 1.')

        logging.info('Begin to generate permuted model 2.')
        permuted_model_2_path = os.path.join(self.args.save_path, 'permuted_model_2.pth')
        self.generate_permuted_model(self.base_models[1], pis_model_2, permuted_model_2_path)
        logging.info('Finish generating permuted model 2.')

        # Synthetic check that fused model indeed got update.
        # for model1_parameter, fused_model_parameter in zip(self.base_models[0].parameters(), self.target_model.parameters()):
        #     print('Difference between weights of model 1 and fused model:', (model1_parameter - fused_model_parameter).abs().mean())

        logging.info('Model fusion for RNN completed.')

    def generate_permuted_model(self, model, pis, path):
        config = model.get_model_config()
        print(config)
        permuted_model = fuse_rnn_models.get_model(self.args.model_name, self.args.input_dim, config, self.args.encoder)
        if torch.cuda.is_available():
            permuted_model.cuda()
        
        for i in range(1, len(pis)):
            prev_pi = pis[i-1]
            cur_pi = pis[i]
            k_l = cur_pi.size(-2)
            k_l_prev = prev_pi.size(-2)

            Ws, Hs = model.get_layer_weights(i)

            # Compute permuted Ws
            Ws_permuted = torch.matmul(cur_pi, torch.matmul(Ws, k_l * k_l_prev * prev_pi.transpose(0, 1)))

            # Compute permuted Hs
            if Hs is not None:
                Hs_permuted = torch.matmul(cur_pi, torch.matmul(Hs, k_l * k_l * cur_pi.transpose(0, 1)))
            else:
                Hs_permuted = None
            
            # Update permuted model
            if self.args.model_name in ['LSTM', 'lstm']:
                permuted_model.update_layer_weights(i, Ws_permuted, Hs_permuted)
            else:
                permuted_model_Ws, permuted_model_Hs = permuted_model.get_layer_weights(i)
                permuted_model_Ws.data = Ws_permuted.data
                if permuted_model_Hs is not None:
                    permuted_model_Hs.data = Hs_permuted.data
        
        # Synthetic check on permuted model
        for parameter_1, parameter_2 in zip(model.parameters(), permuted_model.parameters()):
            print('Difference between the weights of original model and permuted model:', (parameter_1 - parameter_2).abs().mean()) 

        self.save_target_model(permuted_model, path)

    def save_target_model(self, model, path):
        torch.save(
            {
              'model_state_dict': model.state_dict(),
              'config': model.get_model_config()
            },
            path
        )
        logging.info('permuted model saved at {}'.format(path))

    def compute_weights_diff(self, layer):
        logging.info("Layer:{}".format(layer))
        target_Ws, target_Hs = self.target_model.get_layer_weights(layer_num=layer)
        for idx, model in enumerate(self.base_models):
            Ws, Hs = model.get_layer_weights(layer_num=layer)
            diff = (target_Ws - Ws).abs().mean()
            logging.info("Idx: {}, Ws diff: {}".format(idx, diff.detach().cpu()))
            if Hs is not None:
                diff = (target_Hs - Hs).abs().mean()
                logging.info("Idx: {}, Hs diff: {}".format(idx, diff.detach().cpu()))

    def fuse_single_layer(self, layer, prev_pis):
        print("Layer:", layer)
        logging.info('Fusing layer {}'.format(layer))
        self.alpha_h = self.alpha_h_arr[layer - 1]
        logging.info('Current layer alpha:{}'.format(self.alpha_h))

        self.tlp_sinkhorn_reg = self.tlp_sinkhorn_reg_list[layer - 1]
        logging.info('Current layer reg:{}'.format(self.tlp_sinkhorn_reg))
        # Fuses a singe layer of the networks.
        base_Ws = []
        base_Hs = []
        for model in self.base_models:
            Ws, Hs = model.get_layer_weights(layer_num=layer)
            base_Ws.append(Ws)
            base_Hs.append(Hs)
        target_Ws, target_Hs = self.target_model.get_layer_weights(layer_num=layer)
        # TODO: add a new method to get W and H
        # TODO: check how to fuse Embedding layer and how to 
        if layer == 1 and self.args.encoder:
            # Input Embedding layer, the weight should be transposed
            return self.fuse_single_linear_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_Ws, target_Ws, prev_pis)
        elif layer == self.target_model.num_layers:
            # output linear layer (decoder)
            return self.fuse_single_linear_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_Ws, target_Ws, prev_pis)
        else:
            # hidden layer

            pis = self.fuse_single_hidden_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                                       base_Ws, base_Hs, target_Ws, target_Hs, prev_pis)
            if isinstance(self.target_model, (rnn_models.LSTMWithDecoder)):
                self.target_model.update_layer_weights(layer, target_Ws, target_Hs)
            return pis

    def fuse_single_hidden_layer_helper(self, num_models, layer, is_last_layer, base_Ws, base_Hs,
                                        target_Ws, target_Hs, prev_pis):
        # Fuses a singe layer of the networks.
        logging.info('Helper fusing (RNN hidden) layer {}'.format(layer))

        Ws = copy.deepcopy(target_Ws.data)
        Hs = copy.deepcopy(target_Hs.data)
        if torch.cuda.is_available() and self.args.device == 'cuda':
            Ws = Ws.cuda()
            Hs = Hs.cuda()

        if is_last_layer:
            # For the last layer, we know the coupling for the final layer.
            pis = []
            for i in range(num_models):
                n = target_Ws.size(-2)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available() and self.args.device == 'cuda':
                    cur_pi = cur_pi.cuda()
                pis.append(cur_pi)
            Ws, Hs = self.solve_hidden_weights(base_Ws, base_Hs, prev_pis, pis)
            target_Ws.data = Ws.data
            target_Hs.data = Hs.data
            return pis

        # When this is not the last layer, we iteratively optimize
        # for pi and weights.
        # original method to initialize pis for hidden weights
        pis = []
        for W_i in base_Ws:
            n = target_Ws.size(-2) # dimension of current layer in the target model, should be equal to target_Hs.size(0)
            m = W_i.size(-2) # dimension of current layer in base model i, should be equal to  H_i.size(0))
            cur_pi = torch.ones(size=(n, m), dtype=torch.float) / (1.0 * n * m)
            if torch.cuda.is_available() and self.args.device == 'cuda':
                cur_pi = cur_pi.cuda()
            pis.append(cur_pi)

        for pi in pis:
            logging.info("Pi min: {}, Pi max: {}, Pi sum.{}".format(
                pi.min().detach().cpu(), pi.max().detach().cpu(),
                pi.sum().detach().cpu()))

        max_iter = self.niters_rnn
        threshold = 1e-5
        actual_iter = max_iter
        for i in range(1, max_iter + 1):
            print("fuse_single_hidden_layer_helper, iteration", i)
            # print(base_Ws[0].size(), base_Hs[0].size(), target_Ws.size(), target_Hs.size(), prev_pis[0].size(), pis[0].size())
            if self.args.solve_ih_pi_first:
                print("Solve input-to-hidden pi and use it to initialize hidden-to-hidden pi")
                pis = self.solve_pi_linear(base_Ws, Ws, prev_pis, layer == 1)
                # print("Also solve target weight first")
                # target_Ws = self.solve_linear_weights(base_Ws, prev_pis, pis)
            
            new_pi = self.solve_pi_hidden(base_Ws, base_Hs, Ws, Hs, prev_pis, pis, layer == 1)
            new_Ws, new_Hs = self.solve_hidden_weights(base_Ws, base_Hs, prev_pis, new_pi)

            Ws_epsilon = (Ws - new_Ws).pow(2).mean()
            Hs_epsilon = (Hs - new_Hs).pow(2).mean()
            pi_epsilon = 0
            for new_pi_i, pi_i in zip(new_pi, pis):
                pi_epsilon += (new_pi_i - pi_i).pow(2).mean()
            Ws = new_Ws
            Hs = new_Hs
            pis = new_pi
            logging.info('Ws_epsilon {}, Hs_epsilon {}, pi_epsilon {}'.format(Ws_epsilon,
                                                                              Hs_epsilon,
                                                                              pi_epsilon))
            for pi in pis:
                logging.info("Pi min: {}, Pi max: {}, Pi sum.{}".format(
                    pi.min().detach().cpu(), pi.max().detach().cpu(),
                    pi.sum().detach().cpu()))
            obj = self.compute_barcenter_objective(base_Ws, base_Hs, Ws, Hs, prev_pis, pis,
                                                   layer == 1)
            logging.info("Barycenter obj = {0:6g}".format(obj.detach().cpu()))
            if Ws_epsilon < threshold and Hs_epsilon < threshold and pi_epsilon < threshold:
                logging.info('Ws_epsilon {}, Hs_epsilon {}, pi_epsilon {}'.format(Ws_epsilon,
                                                                                  Hs_epsilon,
                                                                                  pi_epsilon))
                actual_iter = i
                break
            if self.tlp_sinkhorn_reg > self.min_reg:
                self.tlp_sinkhorn_reg *= self.reg_gamma  # Multiplicative update

        target_Ws.data = Ws.data
        target_Hs.data = Hs.data
        logging.info('Num of actual iterations={}'.format(actual_iter))
        logging.info('Ws_epsilon {}, Hs_epsilon {}, pi_epsilon {}'.format(Ws_epsilon,
                                                                          Hs_epsilon,
                                                                          pi_epsilon))
        return pis

    def fuse_single_linear_layer_helper(self, num_models, layer, is_last_layer, base_weights,
                                 target_weights, prev_pi):
        logging.info('Helper fusing (linear) layer {}'.format(layer))
        # Fuses a singe layer of the networks.
        weights = copy.deepcopy(target_weights.data)
        # weights = torch.rand_like(target_weights)
        if torch.cuda.is_available() and self.args.device == 'cuda':
            weights = weights.cuda()

        if is_last_layer:
            # For the last layer, we know the coupling for the final layer.
            pis = []
            for i in range(num_models):
                n = target_weights.size(-2)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available() and self.args.device == 'cuda':
                    cur_pi = cur_pi.cuda()
                pis.append(cur_pi)
            print("last layer")
            # print("new pi",pis[0])
            weights = self.solve_linear_weights(base_weights, prev_pi, pis)
            # print("new weights",weights)
            # print("squared diffences new_weights",torch.mean((weights-base_weights[0])**2))
            target_weights.data = weights.data
            return pis

        # When this is not the last layer, we iteratively optimize
        # for pi and weights.
        pis = []
        for cur_weight in base_weights:
            n = target_weights.size(-2)
            m = cur_weight.size(-2)
            cur_pi = torch.ones(size=(n, m), dtype=torch.float) / (1.0 * n * m)
            if torch.cuda.is_available() and self.args.device == 'cuda':
                cur_pi = cur_pi.cuda()
            pis.append(cur_pi)

        max_iter = 100
        threshold = 1e-5
        actual_iter = max_iter
        for i in range(1, max_iter + 1):
            print("fuse_single_linear_layer_helper,Iteration:",i)
            new_pi = self.solve_pi_linear(base_weights, weights, prev_pi, layer == 1)
            new_weights = self.solve_linear_weights(base_weights, prev_pi, new_pi)
            # print("max new pi",torch.max(new_pi[0],-1),torch.max(new_pi[0],0))
            # print("new pi",new_pi[0])
            # print("new_weights",new_weights)
            # print("squared diffences new_weights",torch.mean((new_weights-base_weights[0])**2))
            # print("new weights",new_weights)

            weights_epsilon = (weights - new_weights).pow(2).mean()
            pi_epsilon = 0
            for new_pi_i, pi_i in zip(new_pi, pis):
                pi_epsilon += (new_pi_i - pi_i).pow(2).mean()
            weights = new_weights
            pis = new_pi
            logging.info('weights_epsilon {}, pi_epsilon {}'.format(weights_epsilon, pi_epsilon))
            if weights_epsilon < threshold and pi_epsilon < threshold:
                logging.info('weights_epsilon {}, pi_epsilon {}'.format(weights_epsilon, pi_epsilon))
                actual_iter = i
                break
        target_weights.data = weights.data
        logging.info('Num of actual iterations={}'.format(actual_iter))
        return pis

    def solve_pi_hidden(self, base_Ws, base_Hs, target_W, target_H, prev_pis, cur_pis,
                        is_first_layer=False):
        logging.info("Solving inner pi!")
        error = 1.0
        threshold = 1e-5
        num_iters = 0
        while error >= threshold and num_iters < self.num_pi_iters:
            new_pis = []
            tmp_pis = self.solve_pi_hidden_helper(base_Ws, base_Hs, target_W, target_H,
                                                  prev_pis, cur_pis, is_first_layer)
            error = 0.0
            for cur, next in zip(cur_pis, tmp_pis):
                error += (cur-next).abs().mean().detach().cpu()  # Using abs for min
                new_pis.append((1-self.theta_pi)*cur+self.theta_pi*next)
            cur_pis = new_pis
            num_iters += 1
            logging.info("Inner Loop: Epsilon: {}, num iters:{}".format(error, num_iters))
        return cur_pis

    def solve_pi_hidden_helper(self, base_Ws, base_Hs, target_W, target_H, prev_pis, cur_pis,
                               is_first_layer=False):
        logging.info("Solving pi hidden helper!")
        pis = []
        with torch.no_grad():
            if is_first_layer:
                W_t = target_W.unsqueeze(-2) # dim = (cur_dim_W,1,pre_dim_W)
                H_t = target_H.unsqueeze(-2).unsqueeze(-1) # (cur_dim_H_out,1,cur_dim_H_in) -> (cur_dim_H_out,1,cur_dim_H_in,1)
                for W_i, H_i,cur_pi_i in zip(base_Ws, base_Hs, cur_pis):
                    W_i = W_i.unsqueeze(-3) # dim = (1,cur_dim_Wi,pre_dim_Wi) = (1,cur_dim_Wi,pre_dim_W) in the first layer
                    H_i = H_i.unsqueeze(-3).unsqueeze(-2) # (1, cur_dim_Hi_out, cur_dim_Hi_in) -> (1, cur_dim_Hi_out, 1, cur_dim_Hi_in)
                    # diff_W = (W_t - W_i).pow(2) # dim = (cur_dim_W,cur_dim_Wi,pre_dim_W)
                    #print("H_t.size(),H_i.size(),cur_pi_i.size()",H_t.size(),H_i.size(),cur_pi_i.size())
                    try:
                        # dim = (cur_dim_H_out, cur_dim_Hi_out, cur_dim_H_in, cur_dim_Hi_in)
                        cost = (W_t - W_i).pow(2).sum(-1) + self.alpha_h * ((H_t - H_i).pow(2) * cur_pi_i).sum(-1).sum(-1) #TODO: check how cost should be correctly calculated
                        if len(cost.size()) == 3:
                            cost = cost.sum(0)
                    except RuntimeError as e:
                        error = "{}".format(e)
                        if error.startswith("CUDA out of memory."):
                            # cost = for_loop_cost(H_t, H_i, cur_pi_i,LSTM=(self.args.model_name=='LSTM' or self.args.model_name=='lstm'))
                            if len(H_t.size()) == 5: # LSTM
                                cost = (W_t - W_i).pow(2).sum(-1).sum(0)
                                for i in range(4):
                                    cost += self.alpha_h * memory_safe_methods.get_cost(H_t[i], H_i[i], cur_pi_i)
                            else:
                                cost = (W_t - W_i).pow(2).sum(-1) + self.alpha_h * memory_safe_methods.get_cost(H_t, H_i, cur_pi_i)
                        else:
                            print(error)
                            raise ImportError(e)
                    # Different algorithms for solving Linear equations goes here.
                    new_pi = self.ot_solver(cost)
                    #print("new_pi.size()",new_pi.size())
                    pis.append(new_pi)
                return pis
            else:
                W_t = target_W.unsqueeze(-2).unsqueeze(-1) 
                H_t = target_H.unsqueeze(-2).unsqueeze(-1) # (cur_dim_H_out,1,cur_dim_H_in) -> (4,cur_dim_H_out,1,cur_dim_H_in,1)

                for W_i, H_i, prev_pi_i, cur_pi_i in zip(base_Ws, base_Hs, prev_pis, cur_pis):
                    # difference = target_H-H_i
                    W_i = W_i.unsqueeze(-3).unsqueeze(-2) 
                    H_i = H_i.unsqueeze(-3).unsqueeze(-2) # (1, cur_dim_Hi_out, cur_dim_Hi_in) -> (1, cur_dim_Hi_out, 1, cur_dim_Hi_in)
                    #print("H_t.size(),H_i.size(),cur_pi_i.size()",H_t.size(),H_i.size(),cur_pi_i.size())
                    #print("W_t.size(),W_i.size(),prev_pi_i.size()",W_t.size(),W_i.size(),prev_pi_i.size())
                    try:
                        cost = ((W_t - W_i).pow(2) * prev_pi_i).sum(-1).sum(-1) + self.alpha_h * ((H_t - H_i).pow(2) * cur_pi_i).sum(-1).sum(-1) # (4,hid,hid)
                        if len(cost.size()) == 3:
                            cost = cost.sum(0)
                        #print("cost.size()",cost.size())
                    except RuntimeError as e:
                        error = "{}".format(e)
                        if error.startswith("CUDA out of memory."):
                            if len(H_t.size()) == 4:
                                cost = self.alpha_h * memory_safe_methods.get_cost(H_t, H_i, cur_pi_i) + memory_safe_methods.get_cost(W_t, W_i, prev_pi_i)
                            elif len(H_t.size()) == 5: #LSTM
                                cost = 0
                                for idx_mat in range(4):
                                    cost += self.alpha_h * memory_safe_methods.get_cost(H_t[idx_mat], H_i[idx_mat], cur_pi_i)
                                    cost += memory_safe_methods.get_cost(W_t[idx_mat], W_i[idx_mat], prev_pi_i)
                            else:
                                raise NotImplementedError
                            #print("cost.size()",cost.size())
                        else:
                            print(error)
                            raise ImportError(e)
                    # Different algorithms for solving Linear equations goes here.
                    # print("solve_pi_hidden, NOT is_first_layer",cost)
                    new_pi = self.ot_solver(cost)
                    #print("new_pi.size()",new_pi.size())
                    pis.append(new_pi)
                return pis

    def compute_barcenter_objective(self, base_w, base_h, target_w, target_h, prev_pi, pi,
                                    is_first_layer=False):
        ot_dists = []
        with torch.no_grad():
            if is_first_layer:
                w_t = target_w.unsqueeze(-2)
                h_t = target_h.unsqueeze(-2).unsqueeze(-1)
                for w_i, h_i, pi_i in zip(base_w, base_h, pi):
                    w_i = w_i.unsqueeze(-3)
                    h_i = h_i.unsqueeze(-3).unsqueeze(-2)
                    if len(h_t.size()) == 5:  # LSTM case
                        cost = (w_t - w_i).pow(2).sum(-1).sum(0)
                        for i in range(4):
                            cost += self.alpha_h * memory_safe_methods.get_cost(h_t[i], h_i[i], pi_i)
                    else:
                        cost = (w_t - w_i).pow(2).sum(-1)
                        cost += self.alpha_h * memory_safe_methods.get_cost(h_t, h_i, pi_i)
                    ot_dists.append(torch.sum(cost * pi_i))
            else:
                W_t = target_w.unsqueeze(-2).unsqueeze(-1)
                H_t = target_h.unsqueeze(-2).unsqueeze(-1)
                for W_i, H_i, prev_pi_i, cur_pi_i in zip(base_w, base_h, prev_pi, pi):
                    W_i = W_i.unsqueeze(-3).unsqueeze(-2)
                    H_i = H_i.unsqueeze(-3).unsqueeze(-2)
                    try:
                        cost = ((W_t - W_i).pow(2) * prev_pi_i).sum(-1).sum(-1) + self.alpha_h * (
                                    (H_t - H_i).pow(2) * cur_pi_i).sum(-1).sum(-1)  # (4,hid,hid)
                        if len(cost.size()) == 3:
                            cost = cost.sum(0)
                    except RuntimeError as e:
                        error = "{}".format(e)
                        if error.startswith("CUDA out of memory."):
                            if len(H_t.size()) == 4:
                                cost = self.alpha_h * memory_safe_methods.get_cost(H_t, H_i,
                                                                                   cur_pi_i) + memory_safe_methods.get_cost(
                                    W_t, W_i, prev_pi_i)
                            elif len(H_t.size()) == 5:  # LSTM
                                cost = 0
                                for idx_mat in range(4):
                                    cost += self.alpha_h * memory_safe_methods.get_cost(H_t[idx_mat], H_i[idx_mat],
                                                                                        cur_pi_i)
                                    cost += memory_safe_methods.get_cost(W_t[idx_mat], W_i[idx_mat], prev_pi_i)
                        else:
                            print(error)
                            raise ImportError(e)
                    ot_dists.append(torch.sum(cost * cur_pi_i))
            return sum(ot_dists) / len(ot_dists)
    
    def solve_pi_linear(self, base_weights, target_weights, prev_pi, is_first_layer=False):
        pi = []
        with torch.no_grad():
            if is_first_layer:
                w = target_weights.unsqueeze(-2) #1
                for w_i, prev_pi_i in zip(base_weights, prev_pi):
                    w_i = w_i.unsqueeze(-3) #0
                    cost = (w - w_i).pow(2).sum(dim=-1)
                    if len(cost.size()) ==3:
                        cost = cost.sum(0)
                    print("cost.size()",cost.size())
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi
            else:
                w = target_weights.unsqueeze(-2).unsqueeze(-1) # 1,3

                for w_i, prev_pi_i in zip(base_weights, prev_pi):
                    w_i = w_i.unsqueeze(-3).unsqueeze(-2)
                    try:
                        diff = (w - w_i).pow(2)
                        cost = (diff * prev_pi_i).sum(-1).sum(-1)
                        if len(cost.size()) ==3:
                            cost = cost.sum(0)
                        print("cost.size()",cost.size())
                    except RuntimeError as e:
                        error = "{}".format(e)
                        if error.startswith("CUDA out of memory."):
                            if W_t.size() == 4:
                                cost = memory_safe_methods.get_cost(W_t, W_i, prev_pi_i)
                            elif W_t.size() == 5: #LSTM
                                cost = 0
                                for idx_mat in range(4):
                                    cost += memory_safe_methods.get_cost(W_t[idx_mat], W_i[idx_mat], prev_pi_i)
                            print("cost.size()",cost.size())
                            # cost = memory_safe_methods.get_cost(w, w_i, prev_pi_i)
                            # if len(cost.size()) ==3:
                            #     cost = cost.sum(0)
                        else:
                            print(error)
                            raise ImportError(e)
                    # Different algorithms for solving Linear equations goes here.
                    # print("solve_pi_linear,NOT is_first_layer, torch.diagonal(cost)",torch.argmin(cost,-1),torch.diagonal(cost))
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi

    def solve_hidden_weights(self, base_Ws, base_Hs, prev_pi, pi):
        k_l = pi[0].size(-2)
        k_l_prev = prev_pi[0].size(-2)
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
        k_l = pi[0].size(-2)
        k_l_prev = prev_pi[0].size(-2)
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
                cost = memory_safe_methods.get_activation_cost(target_model_activations, activations)
                # Different algorithms for solving Linear equations can go here.
                cur_pi = self.ot_solver(cost)
                pi.append(cur_pi)
        return pi

    def ot_solver(self, cost):
        if self.args.tlp_ot_solver == 'sinkhorn':
            epsilon = self.tlp_sinkhorn_reg
            pi, _ = ot_algorithms.sinkhorn_coupling(cost, epsilon=epsilon, niter=100)
            return pi
        elif self.args.tlp_ot_solver == 'emd':
            pi, _ = ot_algorithms.ot_emd_coupling(cost)
            return pi
        else:
            raise NotImplementedError



class TLPFusionRNNNoHidden(TLPFusionRNN):
    def __init__(self, args, base_models, target_model, data):
        super().__init__(args, base_models, target_model, data)

    def fuse_single_hidden_layer_helper(self, num_models, layer, is_last_layer, base_Ws,base_Hs, target_Ws,target_Hs, prev_pis):
        # Fuses a singe layer of the networks.
        logging.info('No-Hidden Helper fusing (RNN hidden) layer {}'.format(layer))

        Ws = copy.deepcopy(target_Ws.data)
        Hs = copy.deepcopy(target_Hs.data)
        if torch.cuda.is_available() and self.args.device == 'cuda':
            Ws = Ws.cuda()
            Hs = Hs.cuda()

        if is_last_layer:
            # For the last layer, we know the coupling for the final layer.
            pis = []
            for i in range(num_models):
                n = target_Ws.size(-2)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available() and self.args.device == 'cuda':
                    cur_pi = cur_pi.cuda()
                pis.append(cur_pi)
            Ws, Hs = self.solve_hidden_weights(base_Ws,base_Hs,prev_pis, pis)
            target_Ws.data = Ws.data
            target_Hs.data = Hs.data
            return pis

        # When this is not the last layer, we iteratively optimize
        # for pi and weights.
        pis = []
        for W_i in base_Ws:
            n = target_Ws.size(-2) # dimension of current layer in the target model, should be equal to target_Hs.size(0)
            m = W_i.size(-2) # dimension of current layer in base model i, should be equal to  H_i.size(0))
            cur_pi = torch.ones(size=(n, m), dtype=torch.float) / (1.0 * n * m)
            if torch.cuda.is_available() and self.args.device == 'cuda':
                cur_pi = cur_pi.cuda()
            pis.append(cur_pi)

        max_iter = 100
        threshold = 1e-5
        actual_iter = max_iter
        for i in range(1, max_iter + 1):
            print("no_hidden fuse_single_hidden_layer_helper, iteration",i)
            # print(base_Ws[0].size(), base_Hs[0].size(), target_Ws.size(), target_Hs.size(), prev_pis[0].size(), pis[0].size())
            new_pi = self.solve_pi_linear(base_Ws, target_Ws, prev_pis, layer == 1)
            # Ws_linear = self.solve_linear_weights(base_Ws,prev_pis, new_pi)
            new_Ws, new_Hs = self.solve_hidden_weights(base_Ws, base_Hs, prev_pis, new_pi)

            Ws_epsilon = (Ws - new_Ws).pow(2).mean()
            Hs_epsilon = (Hs - new_Hs).pow(2).mean()
            pi_epsilon = 0
            for new_pi_i, pi_i in zip(new_pi, pis):
                pi_epsilon += (new_pi_i - pi_i).pow(2).mean()
            Ws = new_Ws
            Hs = new_Hs
            pis = new_pi
            if Ws_epsilon < threshold and Hs_epsilon < threshold and pi_epsilon < threshold:
                logging.info('Ws_epsilon {}, Hs_epsilon {}, pi_epsilon {}'.format(Ws_epsilon, Hs_epsilon, pi_epsilon))
                actual_iter = i
                break
        target_Ws.data = Ws.data
        target_Hs.data = Hs.data
        logging.info('Num of actual iterations={}'.format(actual_iter))
        return pis



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

                        
