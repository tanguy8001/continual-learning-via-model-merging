import torch
import torch.nn as nn

from utils import hook


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, bias=False, fc_nodes=512):
        super(VGG, self).__init__()
        self.fc_nodes = fc_nodes  # Nodes in each layer of the dense model.
        self.num_classes = num_classes
        self.features = features
        self.bias = bias
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, fc_nodes, bias=self.bias),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc_nodes, fc_nodes, bias=self.bias),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc_nodes, num_classes, bias=self.bias),
        )
        if init_weights:
            self._initialize_weights()

        self.layers_count = 0
        self.model_layers = []
        for node in self.features._modules.values():
            if isinstance(node, nn.Conv2d):
                self.layers_count += 1
                self.model_layers.append(node)
        self.conv_layers_count = self.layers_count
        for node in self.classifier._modules.values():
            if isinstance(node, nn.Linear):
                self.layers_count += 1
                self.model_layers.append(node)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def num_layers(self):
        return self.layers_count

    @property
    def input_dim(self):
        return 3

    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        return self.model_layers[layer_num - 1].weight

    def get_model_config(self):
        return {'num_classes': self.num_classes}

    def get_activations(self, x, layer_num=0, pre_activations=True):
        # x is generally B x C x ...
        # For activations this is converted to C x B x ... format.
        if layer_num == 0:
            return torch.transpose(x, 0, 1)
        if pre_activations:
            cur_hook = hook.Hook(self.model_layers[layer_num - 1])
        else:
            raise NotImplementedError
        self.eval()
        _ = self.forward(x)
        cur_hook.close()
        return cur_hook.output.transpose(0, 1).detach()


def make_layers(cfg, batch_norm=False, bias=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, bias=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, bias=bias), bias=bias, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def vgg11(pretrained=False, progress=True, bias=False, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        :param bias: If we want a bias in the model
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, bias=bias, **kwargs)


#################### TESTS ########################


def test_vgg11():
    model = vgg11(num_classes=10)
    print(model)


if __name__ == "__main__":
    test_vgg11()
