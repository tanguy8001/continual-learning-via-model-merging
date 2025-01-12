import copy
import logging
import math
import pdb
import torch

from utils import ot_algorithms
import tlp_fusion

def zero_tensor():
    z = torch.tensor([0.0])
    if torch.cuda.is_available():
        return z.cuda()
    return z


class FrankWolfeFusion:
    """
        This class currently assumes that the base_models and the target_model
        has the same model type and differ in the number of nodes per layer

        base_models: List of base models
        target_dims: Target model nodes upper bound
        """

    def __init__(self, args, base_models, target_model, data):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model
        self.data = data

    def fuse(self):
        logging.info('Starting model fusion.')
        if torch.cuda.is_available():
            for model in self.base_models:
                model.cuda()
            if self.data is not None:
                self.data = self.data.cuda()
            prev_pi = []
            input_dim = self.target_model.input_dim
            for i in range(len(self.base_models)):
                pi = torch.eye(input_dim, dtype=torch.float) / (1.0 * input_dim)
                if torch.cuda.is_available():
                    pi = pi.cuda()
                prev_pi.append(pi)

            for i in range(1, self.target_model.num_layers + 1):
                if self.args.fw_single_layer_fusion_type == 'support':
                    cur_pi = self.fuse_single_layer_using_support(i, prev_pi)
                else:
                    cur_pi = self.fuse_single_layer(i, prev_pi)
                prev_pi = cur_pi

            logging.info('Model Fusion completed')

    def fuse_single_layer_using_support(self, layer, prev_pi):
        # This method fuses a single layer using the support obtained from the tlp method.
        logging.info('Fusing layer {} using the support method.'.format(layer))
        base_weights = []
        nu = []  # This is the list of measure on the layers of base models.
        # Obtains the weights from the base models.
        for model in self.base_models:
            w = model.get_layer_weights(layer_num=layer)
            base_weights.append(w.detach())
            nu.append(torch.ones(w.size(0)) / w.size(0))

        def get_support(layer, base_weights, weights, prev_pi):
            # Returns the support using tlp algorithm
            # Modifies the weights tensor internally.
            tlp_args = copy.deepcopy(self.args)
            tlp_args.tlp_ot_solver = 'sinkhorn'
            tlp_args.tlp_sinkhorn_regularization = self.args.fw_sinkhorn_regularization
            tlp_args.tlp_cost_choice = 'weight'
            tlp_fusion_method = tlp_fusion.TLPFusion(tlp_args, None, None, None)
            return tlp_fusion_method.fuse_single_layer_helper(len(self.base_models), layer,
                                                              layer == self.target_model.num_layers,
                                                              base_weights, weights, prev_pi)

        if layer == self.target_model.num_layers:
            # For final layer, the support is chosen as the final weights.
            # This is because the number of nodes are same as required number of logits/classes.
            target_weights = self.target_model.get_layer_weights(layer_num=layer)
            return get_support(layer, base_weights, target_weights, prev_pi)

        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        weights = copy.deepcopy(target_weights.data)
        _ = get_support(layer, base_weights, weights, prev_pi)  # Weights represent the support
        mu = torch.zeros(size=(weights.size(0),))
        if torch.cuda.is_available():
            mu = mu.cuda()
            for i in range(len(nu)):
                nu[i] = nu[i].cuda()
        mu[0] = 1.0  # Initialize the solution with 0th support weight.

        def get_node_costs(w1, w2, pi):
            # Returns cost between w1 and w2.
            # w1 is the first set of weights, w2 is the second set of weights.
            # pi is the coupling between prev layer of these weights.
            if layer == 1:
                return (w1.unsqueeze(1) - w2.unsqueeze(0)).pow(2).sum(-1)
            else:
                w1 = w1.unsqueeze(1).unsqueeze(-1)
                w2 = w2.unsqueeze(0).unsqueeze(-2)
                return ((w1 - w2).pow(2) * pi).sum(-1).sum(-1)

        cost = []  # Costs between support weights and nodes in prev models, [k_l x k_l_i]
        for i in range(len(base_weights)):
            cost_i = []
            for w in weights:
                cost_i.append(get_node_costs(w.unsqueeze(0), base_weights[i], prev_pi[i]))
            cost.append(torch.cat(cost_i, dim=0))
        # Cost between the nodes of support. Assumes identity coupling for this
        # since the support points are each different.
        self_cost = (weights.unsqueeze(0) - weights.unsqueeze(1)).pow(2).sum(-1)

        for idx in range(1, self.target_model.channels[layer]):
            # We add as many nodes as are there in the target model and then prune them later.
            beta = []  # List of beta for OT_epsilon(mu, nu[i])
            epsilon = self.args.fw_sinkhorn_regularization
            nz_idx = mu > 0  # Non zero index
            for i in range(len(base_weights)):
                _, _, _, b = ot_algorithms.sinkhorn_coupling_and_potentials(mu=mu[nz_idx], nu=nu[i],
                                                                            cost=cost[i][nz_idx],
                                                                            epsilon=epsilon,
                                                                            niter=100)
                beta.append(b)

            if idx == 1:
                self_beta = zero_tensor()
            else:
                _, _, _, self_beta = ot_algorithms.sinkhorn_coupling_and_potentials(mu=mu[nz_idx], nu=mu[nz_idx],
                                                                                    cost=self_cost[nz_idx, :][:, nz_idx],
                                                                                    epsilon=epsilon,
                                                                                    niter=100)

            new_weight_idx = self.solve_weights_from_support(base_weights, beta, self_beta, cost,
                                                             self_cost[:, nz_idx])
            mu *= idx / (idx + 2.0)
            mu[new_weight_idx] += 2.0 / (idx + 2.0)

        # Drop the nodes with low associated mass
        low_prob = 0.01 * torch.max(mu)
        select_idx =  mu > low_prob

        mu = mu[select_idx]
        weights = weights[select_idx]
        logging.info("Total support: {}, selected: {}".format(len(select_idx),
                                                              torch.sum(select_idx).item()))
        for i in range(len(cost)):
            cost[i] = cost[i][select_idx]

        # Updates the target model and stats based on pruned weights.
        self.target_model.update_layer(layer=layer, hidden_nodes=mu.size(0), weights=weights)
        cur_pi = []
        epsilon = self.args.fw_sinkhorn_regularization
        for i in range(len(base_weights)):
            pi, _, _, _ = ot_algorithms.sinkhorn_coupling_and_potentials(mu, nu[i], cost[i],
                                                                         epsilon=epsilon,
                                                                         niter=100)
            cur_pi.append(pi)
        return cur_pi

    def fuse_single_layer(self, layer, prev_pi):
        # Fuses single layer using fw algorithm and using no support weights.
        # This is the original version of the algorithm.
        logging.info('Fusing layer {}'.format(layer))
        base_weights = []
        nu = []
        for model in self.base_models:
            w = model.get_layer_weights(layer_num=layer)
            base_weights.append(w.detach())
            nu.append(torch.ones(w.size(0))/w.size(0))

        weights = torch.rand((1, self.target_model.channels[layer-1]))  # Weights are stacked to this
        mu = torch.ones(1)

        if torch.cuda.is_available():
            weights = weights.cuda()
            mu = mu.cuda()
            for i in range(len(nu)):
                nu[i] = nu[i].cuda()

        if layer == self.target_model.num_layers:
            pi = []
            target_weights = self.target_model.get_layer_weights(layer_num=layer)
            for i in range(len(self.base_models)):
                n = target_weights.size(0)
                cur_pi = torch.eye(n, dtype=torch.float) / (1.0 * n)
                if torch.cuda.is_available():
                    cur_pi = cur_pi.cuda()
                pi.append(cur_pi)
            tlp_fusion_dummy = tlp_fusion.TLPFusion(None, None, None, None)
            weights = tlp_fusion_dummy.solve_weights(base_weights, prev_pi, pi)
            target_weights.data = weights.data
            return pi
        cost = []

        def get_node_costs(w1, w2, pi):
            # w1 is the first set of weights, w2 is the second set of weights
            # pi is the coupling between prev layer of these weights
            if layer == 1:
                return (w1.unsqueeze(1) - w2.unsqueeze(0)).pow(2).sum(-1)
            else:
                w1 = w1.unsqueeze(1).unsqueeze(-1)
                w2 = w2.unsqueeze(0).unsqueeze(-2)
                return ((w1 - w2).pow(2) * pi).sum(-1).sum(-1)

        for i in range(len(base_weights)):
            cost.append(get_node_costs(weights, base_weights[i], prev_pi[i]))  # 1 x k_l_i

        for idx in range(1, self.target_model.channels[layer]):
            # Loop to add a new node
            beta = []  # List of beta for OT_epsilon(mu, nu[i])
            epsilon = self.args.fw_sinkhorn_regularization
            for i in range(len(base_weights)):
                _, _, _, b = ot_algorithms.sinkhorn_coupling_and_potentials(mu=mu, nu=nu[i],
                                                                            cost=cost[i],
                                                                            epsilon=epsilon,
                                                                            niter=100)
                beta.append(b)
            self_cost = (weights.unsqueeze(0) - weights.unsqueeze(1)).pow(2).sum(-1)
            if idx == 1:
                self_beta = zero_tensor()
            else:
                _, _, _, self_beta = ot_algorithms.sinkhorn_coupling_and_potentials(mu=mu, nu=mu,
                                                                                    cost=self_cost,
                                                                                    epsilon=epsilon,
                                                                                    niter=100)
            new_weights = self.solve_weights(layer, base_weights, weights, beta, self_beta, prev_pi)
            weights = torch.cat((weights, new_weights), dim=0)
            for i in range(len(base_weights)):
                c = get_node_costs(new_weights, base_weights[i], prev_pi[i])
                cost[i] = torch.cat((cost[i], c), dim=0)
            prob_mass = torch.tensor([2.0/(idx + 2.0)])
            if torch.cuda.is_available():
                prob_mass = prob_mass.cuda()
            mu = torch.cat((mu * idx / (idx + 2.0), prob_mass), dim=0)

        # Drop the nodes with low associated mass
        mu_cum = torch.cumsum(mu.flip(dims=[0]), dim=0).flip(dims=[0])
        low_percentile = 0.005
        low_index = torch.sum(mu_cum > (1-low_percentile))
        # low_index = torch.sum(mu < 0.01 * torch.max(mu))
        mu = mu[low_index:]
        weights = weights[low_index:]
        for i in range(len(cost)):
            cost[i] = cost[i][low_index:]
        self.target_model.update_layer(layer=layer, hidden_nodes=mu.size(0), weights=weights)
        cur_pi = []
        epsilon = self.args.fw_sinkhorn_regularization
        for i in range(len(base_weights)):
            # cur_opt.append(ot_algorithms.ot_emd(mu=mu, nu=nu[i], cost=cost[i]))
            pi, _, _, _ = ot_algorithms.sinkhorn_coupling_and_potentials(mu, nu[i], cost[i],
                                                                         epsilon=epsilon,
                                                                         niter=100)
            cur_pi.append(pi)
        return cur_pi

    def solve_weights_from_support(self, base_weights, beta, self_beta, cost, self_cost):
        # Finds the new index for the node amongst the support index.
        epsilon = self.args.fw_sinkhorn_regularization
        a_i = 1.0 / len(base_weights)
        loss = 0  # Stores the loss for each each support element
        for i in range(len(base_weights)):
            alpha_loss = (beta[i].unsqueeze(0) - cost[i])/epsilon
            alpha_max = torch.max(alpha_loss, dim=1)[0]
            alpha_loss = torch.exp(alpha_loss - alpha_max.unsqueeze(1))
            alpha_loss = -(alpha_max + torch.log(torch.sum(alpha_loss, dim=1)))
            loss += a_i * alpha_loss
        p_loss = (self_beta.unsqueeze(0) - self_cost) / epsilon
        p_max = torch.max(p_loss, dim=1)[0]
        p_loss = torch.exp(p_loss - p_max.unsqueeze(1))
        p_loss = -(p_max + torch.log(torch.sum(p_loss, dim=1)))
        loss -= p_loss
        return torch.argmin(loss)

    def solve_weights(self, layer, base_weights, weights, beta, self_beta, prev_pi):
        # Finds a new node weight.
        epsilon = self.args.fw_sinkhorn_regularization
        # Finds the new support for the new node.
        C_i = []
        for i in range(len(beta)):
            C_i.append(beta[i])
        C = self_beta

        # Solve the optimization problem to obtain new weights
        w = torch.rand((1, weights.size(1)))
        if torch.cuda.is_available():
            w = w.cuda()
        w.requires_grad = True
        # opt = torch.optim.Adam(params=[w], lr=0.01)
        opt = torch.optim.SGD(params=[w], lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
        niter = 100
        threshold = 1e-4

        if self.args.fw_minimization_type == 'pgd':
            # The maximum ball size for pgd would be some factor of the maximum
            # of the dimensions from the base weights.
            bsize = 0
            for bw in base_weights:
                bsize = max(bsize, torch.max(bw.pow(2)))
            bsize = 2 * bsize * math.sqrt(w.size(1))

        for iter in range(niter):
            loss = 0
            a_i = 1.0/len(base_weights)
            for i in range(len(base_weights)):
                inter_cost = ((w.unsqueeze(1).unsqueeze(-1) -
                               base_weights[i].unsqueeze(0).unsqueeze(-2)).pow(2)*prev_pi[i]).sum(-1).sum(-1)
                # Regularization loss
                if self.args.fw_minimization_type == 'reg':
                    loss += 1e-3 * torch.sum(inter_cost) / epsilon
                    # loss += 10 * w.pow(2).sum()
                    # print('Regularization loss {}'.format(torch.mean(inter_cost)))

                alpha_loss = (C_i[i] - inter_cost.squeeze())/epsilon
                alpha_loss_max = torch.max(alpha_loss)
                alpha_loss_sum = torch.sum(torch.exp(alpha_loss - alpha_loss_max))
                loss += -a_i * (alpha_loss_max + torch.log(alpha_loss_sum))

                p_inter_cost = (w.unsqueeze(1) - weights.unsqueeze(0)).pow(2).sum(-1)
                p_loss = (C - p_inter_cost.squeeze())/epsilon
                p_loss_max = torch.max(p_loss)
                p_loss_sum = torch.sum(torch.exp(p_loss - p_loss_max))
                loss += a_i * (p_loss_max + torch.log(p_loss_sum))

                if torch.isnan(loss) or torch.isinf(loss):
                    print('Loss goes to nan or infinity')
                    pdb.set_trace()

            w_old = copy.deepcopy(w.data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            if self.args.fw_minimization_type == 'pgd':
                w.data *= bsize / torch.norm(w.data)
                err = (w_old - w).pow(2).mean()
            else:
                err = w.grad.pow(2).mean()

            # print('Loss:{}, err:{}'.format(loss.item(), err.item()))
            if err < threshold or iter + 1 == niter:
                logging.info("Actual nits: {}, err: {}".format(iter + 1, err.item()))
                break
        return w.detach()


########## TESTS ##############

def test_fw_fusion_fuse_runs():
    # Fast testing of all the methods in TLP Fusion class.
    from src.tlp_model_fusion import model
    input_dim = 10
    output_dim = 10
    hidden_dims = [60, 65, 13]
    hidden_dims_2 = [22, 33, 64]
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
    parser.add_argument('--fw_cost_choice', type=str, default=None)
    parser.add_argument('--fw_sinkhorn_regularization', type=float, default=0.01)
    parser.add_argument('--fw_single_layer_fusion_type', type=str, default=None,
                        choices=[None, 'support'])
    args = parser.parse_args("")
    args.fw_single_layer_fusion_type = 'support'
    fusion = FrankWolfeFusion(args=args, target_model=target_model, base_models=base_models,
                              data=data)
    fusion.fuse()
    print(target_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fw_fusion_fuse_runs()
