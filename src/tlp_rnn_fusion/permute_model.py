import copy
import logging
import pdb
import torch

from rnn_models import RNNWithDecoder, RNNWithEncoderDecoder

# Class to create new NN by permuting the layers randomly.
class PermuteNN:
    def __init__(self, base_model):
        self.base_model = base_model
        self.permuted_model = None
        self.cur_perms = []

    def permute(self):
        self.permuted_model = copy.deepcopy(self.base_model)
        self.cur_perms = []
        # Returns a randomly permuted model
        logging.info("Permuting the model")
        first_layer_weights = self.permuted_model.get_layer_weights(layer_num=1)
        if type(first_layer_weights) == tuple:
            first_layer_weights = first_layer_weights[0]
        prev_perm = torch.range(0, first_layer_weights.size(1) - 1, dtype=torch.long)
        if torch.cuda.is_available():
            prev_perm = prev_perm.cuda()
        self.cur_perms.append(prev_perm)
        for i in range(1, self.permuted_model.num_layers + 1):
            prev_perm = self.permute_single_layer(layer=i, prev_perm=prev_perm)
            self.cur_perms.append(prev_perm)
        logging.info("Permutations done!")
        return self.permuted_model

    def permute_single_layer(self, layer, prev_perm):
        logging.info("Permuting layer {}".format(layer))
        with torch.no_grad():
            weights_tuple = self.permuted_model.get_layer_weights(layer_num=layer)
            is_rnn_layer = type(weights_tuple) == tuple
            if is_rnn_layer:
                k = weights_tuple[0].size(0)
            else:
                k = weights_tuple.size(0)

            if layer == self.permuted_model.num_layers:
                new_perm = torch.range(0, k - 1, dtype=torch.long)
            else:
                new_perm = torch.randperm(k)
            if torch.cuda.is_available():
                new_perm = new_perm.cuda()
            if is_rnn_layer:
                weights_tuple[0].data = weights_tuple[0][:, prev_perm][new_perm]
                if weights_tuple[1] is not None:
                    weights_tuple[1].data = weights_tuple[1][:, new_perm][new_perm]
            else:
                weights_tuple.data = weights_tuple[:, prev_perm][new_perm]
            return new_perm


# def test_permutations_fc():
#     config = [2, 3, 3, 2]
#     base_model = model.FCModel(input_dim=config[0], hidden_dims=config[1:-1], output_dim=config[-1], bias=False)
#     permuteNN = PermuteNN(base_model)
#     permuted_model = permuteNN.permute()
#     for layer in range(1, base_model.num_layers + 1):
#         print("layer={}".format(layer))
#         base_weights = base_model.get_layer_weights(layer_num=layer)
#         permuted_weights = permuted_model.get_layer_weights(layer_num=layer)
#         print(base_weights)
#         print(permuted_weights)
#         print(permuteNN.cur_perms[layer])


def test_permutations_rnn():
    config = [2, 3, 3]
    base_model = RNNWithEncoderDecoder(output_dim=3,input_dim=2,embed_dim=3,hidden_dims=[4,5],hidden_activations=None)
    # base_model = model.ImageRNN(n_inputs=config[0], n_neurons=config[1:-1], n_outputs=config[-1], n_steps=1)
    permuteNN = PermuteNN(base_model)
    permuted_model = permuteNN.permute()
    for layer in range(1, base_model.num_layers + 1):
        print("layer={}".format(layer))
        base_weights, base_hidden = base_model.get_layer_weights(layer_num=layer)
        permuted_weights, permuted_hidden = permuted_model.get_layer_weights(layer_num=layer)
        print(base_weights)
        print(permuted_weights)
        print(permuteNN.cur_perms[layer])
        if base_hidden is not None:
            print(base_hidden)
            print(permuted_hidden)


if __name__ == "__main__":
    # test_permutations_fc()
    test_permutations_rnn()
