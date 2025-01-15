import torch
import torch.nn.functional as F
import numpy as np

from torch import nn

#from src.tlp_model_fusion.utils import hook

# TO DELETE

class FCModel(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, bias=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(FCModel, self).__init__()
        self.bias = bias
        self.channels = [input_dim] + hidden_dims + [output_dim]
        self.layers = []
        #self.layers = nn.ModuleList()
        for idx in range(1, len(self.channels)):
            cur_layer = [nn.Linear(self.channels[idx - 1], self.channels[idx], bias=self.bias)]
            #cur_layer = nn.ModuleList()
            #cur_layer.append(nn.Linear(self.channels[idx - 1], self.channels[idx], bias=self.bias))
            if idx + 1 < len(self.channels):
                cur_layer.append(nn.ReLU()) 
            seq_cur_layer = nn.Sequential(*cur_layer)
            self.layers.append(seq_cur_layer)
        self.layers_aggregated = nn.Sequential(*self.layers)

    def get_model_config(self):
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1]}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers_aggregated(x)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def input_dim(self):
        return self.channels[0]

    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the weights from the linear layer of this model
        return self.layers[layer_num - 1]._modules['0'].weight

    def get_activations(self, x, layer_num=0, pre_activations=False):
        # layer_num ranges from 1 to total layers.
        if layer_num == 0:
            return x.permute(1, 0)
        assert layer_num <= len(self.layers)
        if pre_activations:
            cur_hook = hook.Hook(self.layers[layer_num - 1]._modules['0'])
        else:
            cur_hook = hook.Hook(self.layers[layer_num - 1])
        self.eval()
        _ = self.forward(x)
        cur_hook.close()
        return cur_hook.output.transpose(0, 1).detach()

    def update_layer(self, layer, hidden_nodes, weights):
        # Updates an internal layer with custom hidden nodes and weights
        # Assumes that weights have appropriate dimension hidden_nodes x prev
        assert weights.size(0) == hidden_nodes
        assert weights.size(1) == self.channels[layer - 1]

        # Update current layer
        cur_layers = [nn.Linear(self.channels[layer - 1], hidden_nodes,
                                bias=self.bias)]
        if layer != self.num_layers:
            cur_layers.append(nn.ReLU())
        cur_layers[0].weight.data = weights.data
        self.layers[layer - 1] = nn.Sequential(*cur_layers)
        self.channels[layer] = hidden_nodes

        # Update Next layer
        if layer < self.num_layers:
            next_layer = [nn.Linear(self.channels[layer], self.channels[layer + 1],
                                    bias=self.bias)]
            if layer + 1 != self.num_layers:
                next_layer.append(nn.ReLU())
            self.layers[layer] = nn.Sequential(*next_layer)
        self.layers_aggregated = nn.Sequential(*self.layers)

# This is a basic image rnn model with a single layer
class ImageRNN(nn.Module):
  def __init__(self, n_steps, n_inputs, n_neurons, n_outputs, act_type='tanh',
               step_start=0, bias=False):
               super(ImageRNN, self).__init__()
               self.n_neurons = n_neurons
               self.n_steps = n_steps
               self.n_inputs = n_inputs
               self.n_outputs = n_outputs
               self.act_type = act_type
               self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons[0], bias=bias,
                                       nonlinearity=self.act_type)
               self.FC = nn.Linear(self.n_neurons[0], self.n_outputs, bias=bias)
               self.relu = nn.ReLU()
               self.step_start = step_start
  
  def init_hidden(self, batch_size):
    h0 = torch.zeros((1, batch_size, self.n_neurons[0]))
    if torch.cuda.is_available():
      h0 = h0.cuda()
    return h0
  
  def forward(self, x):
    # Transforms x to dims: (n_steps x batch_size x n_inputs)
    batch_size = x.size(0)
    x = x.permute(1, 0, 2)
    x = x[self.step_start:self.step_start+self.n_steps]
    hidden = self.init_hidden(batch_size)

    # rnn_out => n_steps x batch_size x n_neurons
    # hidden = 1 x batch_size x n_neurons
    rnn_out, hidden = self.basic_rnn(x, hidden)
    out = self.FC(self.relu(hidden.squeeze()))
    return out

  @property
  def num_layers(self):
    return 2

  @property
  def input_dim(self):
    return self.n_inputs
  
  def get_layer_weights(self, layer_num=1):
    assert 0 < layer_num <= self.num_layers
    if layer_num == 1:
      input_hidden_weight = getattr(self.basic_rnn, 'weight_ih_l0')
      hidden_hidden_weight = getattr(self.basic_rnn, 'weight_hh_l0')
      return input_hidden_weight, hidden_hidden_weight
    elif layer_num == 2:
      return self.FC.weight, None
  
  def get_model_config(self):
    return {'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_neurons': self.n_neurons,
            'n_steps': self.n_steps,
            'act_type': self.act_type,
            'step_start': self.step_start}
  
  def get_activations(self, x, layer_num=0, pre_activations=True):
    raise NotImplementedError
  


def test_get_layer_weights_works():
    model_config = [10, 9, 8, 2]
    model = FCModel(input_dim=model_config[0], hidden_dims=model_config[1:-1],
                    output_dim=model_config[-1], bias=False)
    assert len(model.layers) == 3
    for i in range(1, 4):
        weights = model.get_layer_weights(layer_num=i)
        assert weights.size(0) == model_config[i]
        assert weights.size(1) == model_config[i - 1]
    print('Test get_layer_weights OK!')


def test_get_activations_works():
    model_config = [10, 9, 8, 2]
    model = FCModel(input_dim=model_config[0], hidden_dims=model_config[1:-1],
                    output_dim=model_config[-1], bias=False)
    batch_size = 32
    input = torch.rand(batch_size, model_config[0])
    for i in range(0, 4):
        activations = model.get_activations(x=input, layer_num=i)
        assert activations.size(0) == model_config[i]
        assert activations.size(1) == batch_size
    print('Test get_activations OK!')


def test_update_layer_works():
    model_config = [10, 9, 8, 2]
    model = FCModel(input_dim=model_config[0], hidden_dims=model_config[1:-1],
                    output_dim=model_config[-1], bias=False)
    # Update all layers to be +1
    new_config = [10, 11, 12, 3]
    for layer in range(1, len(new_config)):
        w = torch.rand(new_config[layer], new_config[layer - 1])
        model.update_layer(layer, new_config[layer], w)
        batch_size = 32
        model_config[layer] = new_config[layer]
        input = torch.rand(batch_size, model_config[0])
        for i in range(0, 4):
            activations = model.get_activations(x=input, layer_num=i)
            assert activations.size(0) == model_config[i]
            assert activations.size(1) == batch_size
    print('Test update_layer OK!')

def test_rnn_model():
    model = ImageRNN(n_inputs=28, n_neurons=[128], n_steps=14, n_outputs=10)
    print(model)
    model.cuda()
    batch_size = 16
    data = torch.rand(batch_size, 28, 28)
    data = data.cuda()
    output = model(data)
    assert output.size(0) == batch_size
    assert output.size(1) == 10
    print('Image RNN works')


if __name__ == "__main__":
    # test_get_layer_weights_works()
    # test_get_activations_works()
    # test_update_layer_works()
    test_rnn_model()
