import logging
import pdb
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

import torch
import torch.nn as nn

from src.tlp_model_fusion.utils import hook


class RNNWithDecoder(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, output_dim, embed_dim, hidden_dims, hidden_activations, bias=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(RNNWithDecoder, self).__init__()
        self.bias = bias
        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.channels = [embed_dim] + hidden_dims + [output_dim]
        self.hidden_activations = hidden_activations if hidden_activations is not None else ['tanh']*len(hidden_dims)
        self.rnn_layers = []

        for idx in range(1, len(self.channels)-1):
            try:
                cur_hidden_activations = self.hidden_activations[idx]
            except IndexError:
                cur_hidden_activations = 'tanh'

            cur_layer = nn.RNN(input_size =self.channels[idx-1], hidden_size=self.channels[idx],
                               nonlinearity=cur_hidden_activations, bias=self.bias, batch_first=True)
            self.rnn_layers.append(cur_layer)
        self.rnn_layers = nn.ModuleList(self.rnn_layers)

        # output fully connected layer
        self.decoder = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)

    def decode(self, word_vec):
        with torch.no_grad():
            return self.decoder(word_vec) # size ()
        
    def get_model_config(self):
        # TODO: fix the bug here that output_dim is vocab dim, input_dim is embed_dim
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1],
                'hidden_activations':self.hidden_activations,}

    def forward(self, x):
        """[summary]
        Args:
            x (tensor): size(batch_size,seq_len,input_dim)
        Returns:
            [tensor]: (batch_size,seq_len,)
        """     

        # RNN layers
        cur_input = x
        for idx in range(0, self.num_rnn_layers):
            output_i, h_i = self.rnn_layers[idx](cur_input)
            cur_input = output_i
        # decode
        output = self.decoder(cur_input) # cur_input: size(batch_size,seq_len,embed_dim), output size(batch_size,seq_len,output_dim)
        return output

    @property
    def num_rnn_layers(self):
        return len(self.rnn_layers)

    @property
    def num_layers(self):
        return len(self.rnn_layers) + 1

    def get_layer_weights(self, layer_num=1):
        # return a tuple (input-hidden weights, hidden-hidden weights)
        assert 0 < layer_num <= self.num_rnn_layers + 1, ""
        if layer_num == self.num_rnn_layers + 1:
            return self.decoder.weight, None
        else:
            weights = self.rnn_layers[layer_num - 1].all_weights[0]
            return weights[0], weights[1]
    
    def get_activations(self, x, layer_num=0, pre_activations=False):
        # layer_num ranges from 1 to total layers.
        if layer_num == 0:
            return x.permute(1, 0)
        assert layer_num <= self.num_layers
        if layer_num == self.num_layers:
            if pre_activations:
                cur_hook = hook.Hook(self.decoder._modules['0'])
            else:
                cur_hook = hook.Hook(self.decoder)
        else:
            if pre_activations:
                cur_hook = hook.Hook(self.rnn_layers[layer_num - 1]._modules['0'])
            else:
                cur_hook = hook.Hook(self.rnn_layers[layer_num - 1])
        self.eval()
        _ = self.forward(x)
        cur_hook.close()
        # pdb.set_trace()
        if layer_num == self.num_layers:
            return cur_hook.output.transpose(0, 1).detach()
        else:
            # First component of output: batch x seq len x num hidden units
            return cur_hook.output[0].permute(2, 0, 1).detach()


class RNNWithEncoderDecoder(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, output_dim, input_dim, embed_dim, hidden_dims, hidden_activations,
                 bias=False, tie_weights=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(RNNWithEncoderDecoder, self).__init__()
        self.bias = bias
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.channels = [embed_dim] + hidden_dims + [output_dim]
        self.hidden_activations = hidden_activations if hidden_activations is not None else ['tanh']*len(hidden_dims)
        self.rnn_layers = []

        self.encoder = nn.Linear(self.input_dim, embed_dim, bias=self.bias)
        for idx in range(1, len(self.channels)-1):
            try:
                cur_hidden_activations = self.hidden_activations[idx]
            except IndexError:
                cur_hidden_activations = 'tanh'

            cur_layer = nn.RNN(input_size =self.channels[idx-1],hidden_size=self.channels[idx],nonlinearity=cur_hidden_activations,bias=self.bias,batch_first=True)
            self.rnn_layers.append(cur_layer)
        self.rnn_layers = nn.ModuleList(self.rnn_layers)

        # output fully connected layer
        self.decoder = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)
        if tie_weights:
            if hidden_dims[-1] != embed_dim:
                raise ValueError('When using the tied flag, "dimension of the last hidden layer must be the same as the embedding size"')
            self.decoder.weight = self.encoder.weight

    def get_model_config(self):
        # TODO: fix the bug here that output_dim is vocab dim, input_dim is embed_dim
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1],
                'hidden_activations':self.hidden_activations,}

    def forward(self, x):
        """[summary]

        Args:
            x (tensor): size(batch_size,seq_len)

        Returns:
            [tensor]: (batch_size,seq_len,)
        """     
        # encode
        cur_input = self.encoder(x) # -> (batch_size,seq_len,embed_dim)
        # RNN layers
        for idx in range(0, self.num_rnn_layers):
            output_i, h_i = self.rnn_layers[idx](cur_input)
            # output_i, h_i = self.rnn_layers[idx](cur_input,init_hiddens[idx])
            cur_input = output_i
        # decode
        output = self.decoder(cur_input) # cur_input: size(batch_size,seq_len,embed_dim), output size(batch_size,seq_len,output_dim)
        return output

    @property
    def num_rnn_layers(self):
        return len(self.rnn_layers)

    @property
    def num_layers(self):
        return len(self.rnn_layers) + 2

    # @property
    # def input_dim(self):
    #     return self.channels[0]

    def get_layer_weights(self, layer_num=1):
        # return a tuple (input-hidden weights, hidden-hidden weights)
        assert 0 < layer_num <= self.num_rnn_layers + 2, ""
        if layer_num == 1:
            return self.encoder.weight, None
        elif layer_num == self.num_rnn_layers + 2:
            return self.decoder.weight, None
        else:
            weights = self.rnn_layers[layer_num - 2].all_weights[0]
            return (weights[0],weights[1])


class LSTMWithEncoderDecoder(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dims, hidden_activations,bias=False,tie_weights=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(LSTMWithEncoderDecoder, self).__init__()
        self.bias = bias
        self.input_dim = input_dim
        self.channels = [embed_dim] + hidden_dims + [output_dim]
        self.hidden_activations = hidden_activations if hidden_activations is not None else ['tanh']*len(hidden_dims)
        self.rnn_layers = []

        self.encoder = nn.Linear(self.input_dim,self.channels[0],bias=self.bias)
        for idx in range(1, len(self.channels)-1):
            cur_layer = nn.LSTM(input_size =self.channels[idx-1],hidden_size=self.channels[idx],bias=self.bias,batch_first=True)
            self.rnn_layers.append(cur_layer)
        self.rnn_layers = nn.ModuleList(self.rnn_layers)

        # output fully connected layer
        self.decoder = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)

    def get_model_config(self):
        # TODO: fix the bug here that output_dim is vocab dim, input_dim is embed_dim
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1],
                'hidden_activations':self.hidden_activations,}

    def forward(self, x):
        """[summary]

        Args:
            x (tensor): size(batch_size,seq_len)

        Returns:
            [tensor]: (batch_size,seq_len,)
        """     
        # encode
        cur_input = self.encoder(x) # -> (batch_size,seq_len,embed_dim)
        # LSTM layers
        for idx in range(0, self.num_rnn_layers):
            output_i, h_i = self.rnn_layers[idx](cur_input)
            # output_i, h_i = self.rnn_layers[idx](cur_input,init_hiddens[idx])
            cur_input = output_i
        # decode
        # cur_input: size(batch_size,seq_len,embed_dim),
        # output size(batch_size,seq_len,output_dim)
        output = self.decoder(cur_input)
        return output

    @property
    def num_rnn_layers(self):
        return len(self.rnn_layers)

    @property
    def num_layers(self):
        return len(self.rnn_layers) + 2

    def get_layer_weights(self, layer_num=1):
        # return a tuple (input-hidden weights, hidden-hidden weights)
        assert 0 < layer_num <= self.num_layers, ""
        if layer_num == 1:
            return self.encoder.weight, None
        elif layer_num == self.num_layers:
            return self.decoder.weight, None
        else:
            layer = self.rnn_layers[layer_num-2]
            input_dim = self.channels[layer_num-2]
            hid_dim = self.channels[layer_num-1]
            return layer._parameters['weight_ih_l0'].view(4,hid_dim,input_dim), layer._parameters['weight_hh_l0'].view(4,hid_dim,hid_dim) # returen 4 concatenated weights


class LSTMWithDecoder(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, output_dim, embed_dim, hidden_dims, hidden_activations,bias=False,tie_weights=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(LSTMWithDecoder, self).__init__()
        self.bias = bias
        self.input_dim = embed_dim #no embedding layer
        self.channels = [embed_dim] + hidden_dims + [output_dim]
        self.hidden_activations = hidden_activations if hidden_activations is not None else ['tanh']*len(hidden_dims)
        self.rnn_layers = []

        for idx in range(1, len(self.channels)-1):
            cur_layer = nn.LSTM(input_size =self.channels[idx-1],hidden_size=self.channels[idx],bias=self.bias,batch_first=True)
            self.rnn_layers.append(cur_layer)
        self.rnn_layers = nn.ModuleList(self.rnn_layers)

        # output fully connected layer
        self.decoder = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)

        
    def get_model_config(self):
        # TODO: fix the bug here that output_dim is vocab dim, input_dim is embed_dim
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1],
                'hidden_activations':self.hidden_activations,}

    def forward(self, x):
        """[summary]

        Args:
            x (tensor): size(batch_size,seq_len)

        Returns:
            [tensor]: (batch_size,seq_len,)
        """    
        cur_input = x
        # LSTM layers
        for idx in range(0, self.num_rnn_layers):
            output_i, h_i = self.rnn_layers[idx](cur_input)
            # output_i, h_i = self.rnn_layers[idx](cur_input,init_hiddens[idx])
            cur_input = output_i
        # decode
        output = self.decoder(cur_input) # cur_input: size(batch_size,seq_len,embed_dim), output size(batch_size,seq_len,output_dim)
        return output

    @property
    def num_rnn_layers(self):
        return len(self.rnn_layers)

    @property
    def num_layers(self):
        return len(self.rnn_layers) + 1

    def get_layer_weights(self, layer_num=1):
    # return a tuple (input-hidden weights, hidden-hidden weights)
        assert 0 < layer_num <= self.num_layers, ""
        if layer_num == self.num_layers:
            return self.decoder.weight, None
        else:
            layer = self.rnn_layers[layer_num-1]
            input_dim = self.channels[layer_num-1]
            hid_dim = self.channels[layer_num]
            ihw = getattr(layer, 'weight_ih_l0')
            hhw = getattr(layer, 'weight_hh_l0')
            return ihw.view(4, hid_dim, input_dim), hhw.view(4, hid_dim, hid_dim)
            #return layer._parameters['weight_ih_l0'].view(4,hid_dim,input_dim),layer._parameters['weight_hh_l0'].view(4,hid_dim,hid_dim)  # returen 4 concatenated weights

    def update_layer_weights(self, layer_num, ih_w, hh_w):
        assert  0 < layer_num <= self.num_layers
        logging.info("Updating weights for layer {}".format(layer_num))
        if layer_num == self.num_layers:
            self.decoder.weight.data = ih_w.data
        else:
            layer = self.rnn_layers[layer_num - 1]
            input_dim = self.channels[layer_num - 1]
            hid_dim = self.channels[layer_num]
            ihw = getattr(layer, 'weight_ih_l0')
            hhw = getattr(layer, 'weight_hh_l0')
            ihw.data = ih_w.view(4*hid_dim, input_dim).data
            hhw.data = hh_w.view(4*hid_dim, hid_dim).data

"""
for testing purpose
"""
if __name__ == "__main__":
    seq_len = 3
    embed_dim = 2
    model = RNNWithDecoder(output_dim=5, embed_dim=embed_dim, hidden_dims=[4],
                           hidden_activations=None)
    batch = 7
    x = torch.randn(batch, seq_len, embed_dim)
    output = model(x)
    act = model.get_activations(x, layer_num=1, pre_activations=False)
    print(act.size())
    # pdb.set_trace()
    print(output.size())

