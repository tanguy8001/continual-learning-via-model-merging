import logging
import torch

from utils import memory_safe_methods
from utils import ot_algorithms


class OTFusion:
    """
    This class implements the ot fusion technique mentioned in following paper:
    https://arxiv.org/abs/1910.05653
    The method is based on alignment of layers to one of the pre-initialized
    target model.
    """

    def __init__(self, args, base_models, target_model, data):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model
        self.data = data

    def fuse(self):
        logging.info("Starting model fusion")
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
            prev_pi = cur_pi
        logging.info('Model fusion completed.')

    def fuse_single_layer_helper(self, num_models, layer, is_last_layer, base_weights,
                                 target_weights, prev_pi, prev_similar_pi=None):
        logging.info('Helper Fusing layer {}'.format(layer))
        beta_prev = target_weights.size(1)
        beta = target_weights.size(0)
        is_conv = len(target_weights.size()) == 4
        for i in range(len(base_weights)):
            if is_conv:
                base_weights[i] = torch.matmul(base_weights[i].permute(2, 3, 0, 1),
                                               prev_pi[i] * beta_prev)
                base_weights[i] = base_weights[i].permute(2, 3, 0, 1)
            else:
                base_weights[i] = torch.matmul(base_weights[i], prev_pi[i] * beta_prev)

        if is_last_layer:
            pi = []
            # For last layer, coupling is provided by identity coupling.
            for i in range(num_models):
                n = target_weights.size(0)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available():
                    cur_pi = cur_pi.cuda()
                pi.append(cur_pi)
        else:
            if prev_similar_pi is None:
                if self.args.ad_hoc_cost_choice == 'activation':
                    pi = self.get_activation_coupling(layer=layer)
                else:
                    pi = self.get_weights_coupling(base_weights, target_weights, prev_pi, layer == 1)
            else:
                pi = prev_similar_pi

        for i in range(len(base_weights)):
            if is_conv:
                base_weights[i] = torch.matmul(beta * torch.transpose(pi[i], 0, 1),
                                               base_weights[i].permute(2, 3, 0, 1))
                base_weights[i] = base_weights[i].permute(2, 3, 0, 1)
            else:
                base_weights[i] = torch.matmul(beta * torch.transpose(pi[i], 0, 1),
                                               base_weights[i])

        weights = None
        if 'model_weights' not in self.args.__dict__ or self.args.model_weights is None:
            model_weights = [1.0 / len(pi)] * len(pi)
        else:
            model_weights = self.args.model_weights
        for i, base_weight in enumerate(base_weights):
            if weights is None:
                weights = model_weights[i] * base_weight
            else:
                weights += model_weights[i] * base_weight
        target_weights.data = weights.data
        return pi

    def fuse_single_layer(self, layer, prev_pi):
        logging.info('Fusing layer {}'.format(layer))
        base_weights = []
        for model in self.base_models:
            base_weights.append(model.get_layer_weights(layer_num=layer))
        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        return self.fuse_single_layer_helper(len(self.base_models), layer, layer == self.target_model.num_layers,
                                             base_weights, target_weights, prev_pi)

    def get_weights_coupling(self, base_weights, target_weights, prev_pi,
                             is_first_layer=False):
        # Coupling from base weights to target_weights.
        pi = []
        is_conv = len(target_weights.size()) == 4
        with torch.no_grad():
            w = target_weights.unsqueeze(0)
            for w_i, prev_pi_i in zip(base_weights, prev_pi):
                w_i = w_i.unsqueeze(1)
                try:
                    cost = (w - w_i).pow(2).sum(dim=-1)
                except RuntimeError as e:
                    error = "{}".format(e)
                    if error.startswith("CUDA out of memory."):
                        cost_arr = []
                        for w_i_row in w_i:
                            row_cost = (w - w_i_row.unsqueeze(0)).pow(2).sum(dim=-1)
                            cost_arr.append(row_cost)
                        cost = torch.cat(cost_arr, dim=0)
                    else:
                        print(error)
                        raise ImportError(e)
                if is_conv:
                    cost = cost.sum(-1).sum(-1)
                # Different algorithms for solving Linear equations goes here.
                new_pi = self.ot_solver(cost)
                pi.append(new_pi)
            return pi

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
                cost = memory_safe_methods.get_activation_cost(activations, target_model_activations)
                # Different algorithms for solving Linear equations can go here.
                cur_pi = self.ot_solver(cost)
                pi.append(cur_pi)
        return pi

    def ot_solver(self, cost):
        if self.args.ad_hoc_ot_solver == 'sinkhorn':
            epsilon = self.args.ad_hoc_sinkhorn_regularization
            pi, _ = ot_algorithms.sinkhorn_coupling(cost, epsilon=epsilon, niter=100)
            return pi
        elif self.args.ad_hoc_ot_solver == 'emd':
            pi, _ = ot_algorithms.ot_emd_coupling(cost)
            print(f"The transport map is: {pi}")
            return pi
        else:
            raise NotImplementedError


########## TESTS ############

def test_ad_hoc_fusion_fuse_runs():
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
    parser.add_argument('--ad_hoc_cost_choice', type=str, default=None)
    parser.add_argument('--ad_hoc_ot_solver', type=str, default='sinkhorn')
    parser.add_argument('--ad_hoc_sinkhorn_regularization', type=float, default=0.1)
    args = parser.parse_args("")

    fusion = OTFusion(args=args, target_model=target_model, base_models=base_models,
                      data=data)
    fusion.fuse()


def test_ad_hoc_fusion_fuse_runs_for_resnet():
    # Fast testing of all the methods in TLP Fusion class.
    from src.tlp_model_fusion import resnet_models
    target_model = resnet_models.resnet18(num_classes=10)
    base_models = []
    num_models = 2
    for i in range(num_models):
        new_model = resnet_models.resnet18(num_classes=10)
        base_models.append(new_model)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ad_hoc_cost_choice', type=str, default=None)
    parser.add_argument('--ad_hoc_ot_solver', type=str, default='sinkhorn')
    parser.add_argument('--ad_hoc_sinkhorn_regularization', type=float, default=0.1)
    parser.add_argument('--resnet_skip_connection_handling', type=str, default=['pre'],
                        choices=['pre', 'post'],
                        help='Pre means use pis from previously similar layer, post means handle later')
    args = parser.parse_args("")

    args.resnet_skip_connection_handling = 'pre'
    fusion = OTFusionResnet(args=args, target_model=target_model, base_models=base_models,
                            data=None)
    fusion.fuse()

    args.resnet_skip_connection_handling = 'post'
    target_model = resnet_models.resnet18(num_classes=10)
    fusion = OTFusionResnet(args=args, target_model=target_model, base_models=base_models,
                            data=None)
    fusion.fuse()

    print('Ad hoc OT fusion runs for Resnet')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ad_hoc_fusion_fuse_runs()
    test_ad_hoc_fusion_fuse_runs_for_resnet()