import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.fc2 = nn.Linear(400, 10, bias=False)
        self.enable_dropout = False

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class MlpNetBase(nn.Module):
    def __init__(self, input_dim = 784, num_classes=10, width_ratio=-1):
        super().__init__()
        self.input_dim = input_dim

        disable_bias = True
        enable_dropout = False
        num_hidden_nodes1 = 400
        num_hidden_nodes2 = 200
        num_hidden_nodes3 = 100

        self.width_ratio = width_ratio if width_ratio != -1 else 1

        self.fc1 = nn.Linear(
            input_dim, int(num_hidden_nodes1 / self.width_ratio), bias=not disable_bias
        )
        self.fc2 = nn.Linear(
            int(num_hidden_nodes1 / self.width_ratio),
            int(num_hidden_nodes2 / self.width_ratio),
            bias=not disable_bias,
        )
        self.fc3 = nn.Linear(
            int(num_hidden_nodes2 / self.width_ratio),
            int(num_hidden_nodes3 / self.width_ratio),
            bias=not disable_bias,
        )
        self.fc4 = nn.Linear(
            int(num_hidden_nodes3 / self.width_ratio),
            num_classes,
            bias=not disable_bias,
        )
        self.enable_dropout = enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x)
        x = self.fc4(x)

        return F.log_softmax(x,dim=0)
