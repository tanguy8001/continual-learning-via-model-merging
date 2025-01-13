import copy
import logging
import pdb
import torch
import numpy as np

from utils import ot_algorithms
from utils import memory_safe_methods


class TLPFusion:
    """
    This class currently assumes that the base_models and the target_model
    has the same model type (same number of layer counts).

    base_models: List of base models
    target_model: Target model
    """

    def __init__(self, args, base_models, target_model, data):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model
        self.data = data

    def fuse(self):
        print(self.args.tlp_ot_solver)
        logging.info('Starting model fusion.')
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
            print('total number of couplings for input to hidden:', torch.count_nonzero(cur_pi[0]))
            print('total number of couplings for hidden to hidden:', torch.count_nonzero(cur_pi[1]))
            if self.args.tlp_cost_choice == 'activation':
                prev_pi = self.get_activation_coupling(layer=i)
            else:
                prev_pi = cur_pi  # Other logic follows here
        logging.info('Model fusion completed.')

    def fuse_single_layer_helper(self, num_models, layer, is_last_layer, base_weights,
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
            weights = self.solve_weights(base_weights, prev_pi, pi)
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
            new_weights = self.solve_weights(base_weights, prev_pi, new_pi)

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

    def fuse_single_layer(self, layer, prev_pi):
        logging.info('Fusing layer {}'.format(layer))
        # Fuses a singe layer of the networks.
        base_weights = []
        for model in self.base_models:
            base_weights.append(model.get_layer_weights(layer_num=layer))
        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        return self.fuse_single_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_weights, target_weights, prev_pi)

    def solve_pi(self, base_weights, target_weights, prev_pi, is_first_layer=False):
        pi = []
        is_conv = len(target_weights.size()) == 4
        with torch.no_grad():
            if is_first_layer:
                w = target_weights.unsqueeze(1)
                for w_i, prev_pi_i in zip(base_weights, prev_pi):
                    w_i = w_i.unsqueeze(0)
                    cost = (w - w_i).pow(2).sum(dim=-1)
                    if is_conv:
                        cost = cost.sum(-1).sum(-1)
                    # Different algorithms for solving Linear equations goes here.
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi
            else:
                w = target_weights.unsqueeze(1).unsqueeze(3)

                def get_cost_using_loop(w, w_i, prev_pi_i):
                    # Obtains the cost using loop in w
                    cost_arr = []
                    for w_item in w:
                        w_item = w_item.unsqueeze(0)
                        diff = (w_item - w_i).pow(2)
                        if is_conv:
                            diff = diff.sum(-1).sum(-1)
                        cost_arr.append((diff * prev_pi_i).sum(-1).sum(-1))
                    return torch.cat(cost_arr, dim=0)

                use_loop = False
                for w_i, prev_pi_i in zip(base_weights, prev_pi):
                    w_i = w_i.unsqueeze(0).unsqueeze(2)
                    if use_loop:
                        # cost = get_cost_using_loop(w, w_i, prev_pi_i)
                        cost = memory_safe_methods.get_cost(w, w_i, prev_pi_i)
                    else:
                        # Directly use the memory-safe method for computing the cost
                        cost = memory_safe_methods.get_cost(w, w_i, prev_pi_i)
                        use_loop = True

                    # Different algorithms for solving Linear equations goes here.
                    new_pi = self.ot_solver(cost)
                    pi.append(new_pi)
                return pi

    def solve_weights(self, base_weights, prev_pi, pi):
        k_l = pi[0].size(0)
        k_l_prev = prev_pi[0].size(0)
        weights = None
        if 'model_weights' not in self.args.__dict__ or self.args.model_weights is None:
            model_weights = [1.0 / len(pi)] * len(pi)
        else:
            model_weights = self.args.model_weights

        is_conv = len(base_weights[0].size()) == 4
        for i, (pi_i, prev_pi_i, w_i) in enumerate(zip(pi, prev_pi, base_weights)):
            if is_conv:
                w_i = w_i.permute(2, 3, 0, 1)
                weights_i = torch.matmul(pi_i, torch.matmul(w_i, k_l * k_l_prev * prev_pi_i.transpose(0, 1)))
                weights_i = weights_i.permute(2, 3, 0, 1)
            else:
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


class TLPFusionVGG(TLPFusion):
    """
    Handles the TLP Fusion of VGG network.
    For VGG networks, one needs to separately handle the case of first linear layer
    in the classifier head.
    Currently the code handles only same VGG network architectures.
    """
    def __init__(self, args, base_models, target_model, data):
        super().__init__(args, base_models, target_model, data)
        self.is_linear_layer = False

    def fuse_single_layer(self, layer, prev_pi):
        """
        This method is overridden to handle the case of first linear layer in
        VGG networks. The is done since the first linear layer comes after
        Adaptive avg pooling layer whose output is (7x7).
        """
        logging.info('Fusing layer {}'.format(layer))
        # Fuses a singe layer of the networks.
        base_weights = []
        for model in self.base_models:
            base_weights.append(model.get_layer_weights(layer_num=layer))
        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        logging.info('Target weight dimensions {}'.format(str(target_weights.size())))
        if not self.is_linear_layer and len(target_weights.size()) == 2:
            # To make use of the existing code, we convert the first linear
            # layer weights to cxdx7x7 format. This is to make use of the prev_pi.
            logging.info('First linear layer for VGG')
            self.is_linear_layer = True
            for i in range(len(base_weights)):
                base_weights[i] = base_weights[i].view((base_weights[i].size(0), -1, 7, 7))
            target_weights = target_weights.view((target_weights.size(0), -1, 7, 7))

        return self.fuse_single_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_weights, target_weights, prev_pi)


class TLPFusionResnet(TLPFusion):
    """
    Handles the TLP Fusion of Resnet network.
    For Resnet network we handle the case of skip connections using the prev_pi
    of the layers from which skip connections are present.
    Only works for Resnet type Models.
    """
    def __init__(self, args, base_models, target_model, data):
        super().__init__(args, base_models, target_model, data)
        self.prev_pi_list = []

    def fuse_single_layer(self, layer, prev_pi):
        """
        Handles the skip connection by combining the prev_pi of all the skip connection
        layers for Resnets.
        """
        logging.info('Fusing layer {}'.format(layer))
        if layer == 1:
            # For the first layer the prev_pi would not have been added to list.
            self.prev_pi_list.append(prev_pi)
        # Fuses a singe layer of the networks.
        base_weights = []
        for model in self.base_models:
            w = model.get_layer_weights(layer_num=layer)
            base_weights.append(w)
        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        prev_layers_list = self.target_model.get_prev_layers(layer_num=layer)
        if self.args.resnet_skip_connection_handling == 'pre':
            prev_similar_layer = self.target_model.get_prev_similar_layer(layer_num=layer)
            if prev_similar_layer is None:
                cur_pi = self.fuse_single_layer_helper(len(self.base_models), layer,
                                                       layer == self.target_model.num_layers,
                                                       base_weights, target_weights,
                                                       self.prev_pi_list[prev_layers_list[0]])
            else:
                cur_pi = self.prev_pi_list[prev_similar_layer]
                new_weights = self.solve_weights(base_weights,
                                                 self.prev_pi_list[prev_layers_list[0]],
                                                 cur_pi)
                target_weights.data = new_weights.data
        else:
            all_prev_pi = []
            for base_idx in range(len(prev_pi)):
                sum_prev_pi = 0
                for idx in prev_layers_list:
                    sum_prev_pi += self.prev_pi_list[idx][base_idx]
                all_prev_pi.append(sum_prev_pi)
            logging.info('Target weight dimensions {}'.format(str(target_weights.size())))
            cur_pi = self.fuse_single_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                                   base_weights, target_weights, all_prev_pi)

        if self.args.tlp_cost_choice == 'activation':
            cur_pi = self.get_activation_coupling(layer=layer)
        self.prev_pi_list.append(cur_pi)
        return cur_pi

class TLPFusionRNN(TLPFusion):
  """
  Handles the TLP Fusion of RNN units;
  For RNN units, the fusion happens using the input to hidden weights;
  The hidden to hidden weights need to be adjusted according to the couplings.
  """

  def __init__(self, args, base_models, target_model, data):
    super().__init__(args, base_models, target_model, data)

  def fuse_single_layer(self, layer, prev_pi):
    logging.info('Fusing layer {}'.format(layer))
    # Fuses a single layer of the networks.
    base_input_weights_arr = []
    base_hidden_weights_arr = []
    for model in self.base_models:
      base_input_weights, base_hidden_weights = model.get_layer_weights(layer_num = layer)
      base_input_weights_arr.append(base_input_weights)
      base_hidden_weights_arr.append(base_hidden_weights)
    target_weights, target_hidden_weights = self.target_model.get_layer_weights(layer_num = layer)
    pi = self.fuse_single_layer_helper(len(self.base_models), layer,
                                       layer == self.target_model.num_layers,
                                       base_input_weights_arr, target_weights, prev_pi)
    if target_hidden_weights is not None:
      # The input to hidden weights coupling is applied to orient the 
      # hidden to hidden layer weights.
      weights = None
      if 'model_weights' not in self.args.__dict__ or self.args.model_weights is None:
        model_weights = [1.0 / len(pi)] * len(pi)
      else:
        model_weights = self.args.model_weights
      for i, (pi_i, w_i) in enumerate(zip(pi, base_hidden_weights_arr)):
        k_l = pi_i.size(0)
        weights_i = torch.matmul(pi_i, torch.matmul(w_i, k_l * k_l * pi_i.transpose(0, 1)))
        weights_i *= model_weights[i]
        if weights is None:
          weights = weights_i
        else:
          weights += weights_i
      target_hidden_weights.data = weights
    return pi





########## TESTS ############

def test_tlp_fusion_fuse_runs():
    # Fast testing of all the methods in TLP Fusion class.
    from src.tlp_model_fusion import model
    input_dim = 10
    output_dim = 10
    hidden_dims = [11, 12, 13]
    hidden_dims_2 = [12, 13, 14]
    sample_size = 8

    target_model = model.FCModel(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    base_models = []
    num_models = 5
    for i in range(num_models):
        new_model = model.FCModel(input_dim=input_dim, hidden_dims=hidden_dims if i % 2 == 0 else hidden_dims_2,
                                  output_dim=output_dim)
        base_models.append(new_model)
    data = torch.rand(sample_size, input_dim)

    if torch.cuda.is_available():
        data = data.cuda()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tlp_cost_choice', type=str, default='weight',
                        choices=['weight', 'activation'])
    parser.add_argument('--tlp_ot_solver', type=str, default='sinkhorn',
                        choices=['sinkhorn'])
    parser.add_argument('--tlp_sinkhorn_regularization', type=float, default=0.001)
    parser.add_argument('--tlp_init_type', type=str, default=None,
                        choices=[None, 'identity', 'distill'])
    parser.add_argument('--tlp_init_model', type=int, default=None)

    args = parser.parse_args("")
    fusion = TLPFusion(args=args, target_model=target_model, base_models=base_models,
                       data=data)
    fusion.fuse()


def test_tlp_fusion_vgg_fuse_runs():
    # Fast testing of all the methods in TLP Fusion class.
    from src.tlp_model_fusion import vgg_models
    output_dim = 10
    target_model = vgg_models.vgg11(num_classes=output_dim)
    base_models = []
    num_models = 2
    for i in range(num_models):
        new_model = vgg_models.vgg11(num_classes=output_dim)
        base_models.append(new_model)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tlp_cost_choice', type=str, default='weight',
                        choices=['weight', 'activation'])
    parser.add_argument('--tlp_ot_solver', type=str, default='sinkhorn',
                        choices=['sinkhorn'])
    parser.add_argument('--tlp_sinkhorn_regularization', type=float, default=0.001)
    parser.add_argument('--tlp_init_type', type=str, default=None,
                        choices=[None, 'identity', 'distill'])
    parser.add_argument('--tlp_init_model', type=int, default=None)
    args = parser.parse_args("")

    fusion = TLPFusionVGG(args=args, target_model=target_model, base_models=base_models,
                          data=None)
    fusion.fuse()
    print('VGG Fusion runs!')


def test_tlp_fusion_resnet_fuse_runs():
    # Fast testing of all the methods in TLP Fusion class.
    from src.tlp_model_fusion import resnet_models
    output_dim = 10
    target_model = resnet_models.resnet18(num_classes=output_dim)
    base_models = []
    num_models = 2
    for i in range(num_models):
        new_model = resnet_models.resnet18(num_classes=output_dim)
        base_models.append(new_model)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tlp_cost_choice', type=str, default='weight',
                        choices=['weight', 'activation'])
    parser.add_argument('--tlp_ot_solver', type=str, default='sinkhorn',
                        choices=['sinkhorn'])
    parser.add_argument('--tlp_sinkhorn_regularization', type=float, default=0.001)
    parser.add_argument('--tlp_init_type', type=str, default=None,
                        choices=[None, 'identity', 'distill'])
    parser.add_argument('--tlp_init_model', type=int, default=None)
    parser.add_argument('--resnet_skip_connection_handling', type=str, default=['pre'],
                        choices=['pre', 'post'],
                        help='Pre means use pis from previously similar layer, post means handle later')

    args = parser.parse_args("")
    args.resnet_skip_connection_handling = 'pre'
    fusion = TLPFusionResnet(args=args, target_model=target_model, base_models=base_models,
                             data=None)
    fusion.fuse()

    args.resnet_skip_connection_handling = 'post'
    target_model = resnet_models.resnet18(num_classes=10)
    fusion = TLPFusionResnet(args=args, target_model=target_model, base_models=base_models,
                             data=None)
    fusion.fuse()

    print('Resnet Fusion runs!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_tlp_fusion_fuse_runs()
    test_tlp_fusion_vgg_fuse_runs()
    test_tlp_fusion_resnet_fuse_runs()
