import torch
import torch.nn as nn

from utils import hook


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_batch_norm=False):
        super(BasicBlock, self).__init__()
        self.use_batch_norm=use_batch_norm
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.use_batch_norm:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.use_batch_norm:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_batch_norm=False, bias=False,
                 use_max_pool=True):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.use_batch_norm=use_batch_norm
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_max_pool = use_max_pool

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if use_max_pool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_batch_norm:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=bias)

        # Next few lines collects all the trainable layers and build the skip connection graph.
        # Assumption is that batch normalization is off for current implementation.
        self.all_layers = [self.conv1]  # List of all the layers which have trainable params
        self.prev_layers_list = [[], [0]]  # List of previous layers connected to the current layer
        layer_idx = 1  # layer index starts from 1
        prev_block_downsampled = False
        for cur_layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in cur_layer._modules.values():
                # Each block consists of Conv2d layers in the following order:
                # 1. conv1, conv2, downsample (optional)
                # The optional downsample from previous blocks has to be detected.
                block_dict = block._modules

                # Adding conv1 first.
                layer_idx += 1
                self.all_layers.append(block_dict['conv1'])
                if layer_idx > 3:
                    if prev_block_downsampled:
                        prev_layers = [layer_idx - 2, layer_idx - 1]  # First the main conv layer
                    else:
                        prev_layers = [layer_idx - 1]
                        prev_layers.extend(self.prev_layers_list[layer_idx - 2])
                else:
                    prev_layers = [layer_idx - 1]
                self.prev_layers_list.append(prev_layers)

                # Adding conv2 next.
                layer_idx += 1
                self.all_layers.append(block_dict['conv2'])
                self.prev_layers_list.append([layer_idx - 1])

                has_downsample = False
                if 'downsample' in block_dict.keys():
                    # Adding downsample layer if present
                    has_downsample = True
                    layer_idx += 1
                    self.all_layers.append(block_dict['downsample']._modules['0'])
                    # Sample prev layers as the conv1
                    self.prev_layers_list.append(self.prev_layers_list[layer_idx - 2])
                prev_block_downsampled = has_downsample
        layer_idx += 1
        self.all_layers.append(self.fc)
        if prev_block_downsampled:
            self.prev_layers_list.append([layer_idx - 1, layer_idx - 2])
        else:
            self.prev_layers_list.append([layer_idx - 1])
            self.prev_layers_list[-1].extend(self.prev_layers_list[layer_idx - 2])
        self.prev_similar_layer = [None] * (1 + len(self.all_layers))
        for skip_group in self.prev_layers_list:
            if len(skip_group) > 1:
                first_layer = min(skip_group)
                for other_layer in skip_group:
                    if other_layer != first_layer:
                        if self.prev_similar_layer[other_layer] is None:
                            self.prev_similar_layer[other_layer] = first_layer
        self.pre_activation_layers = self.all_layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.use_batch_norm:
            norm_layer = self._norm_layer
        else:
            norm_layer = None
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_batch_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.use_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def num_layers(self):
        return len(self.all_layers)

    @property
    def input_dim(self):
        return 3

    def get_layer_weights(self, layer_num=1):
        # Returns the layer weights and and previous layers to which it has connection.
        assert 0 < layer_num <= self.num_layers
        return self.all_layers[layer_num - 1].weight

    def get_prev_layers(self, layer_num=1):
        # Returns the prev layer indices which have connection to this layer
        return self.prev_layers_list[layer_num]

    def get_prev_similar_layer(self, layer_num=1):
        # Returns a prev layer which should have similar alignment as the nodes of this layer
        # because they are part of same skip connection group in the resnet model.
        return self.prev_similar_layer[layer_num]

    def get_model_config(self):
        return {'num_classes': self.num_classes,
                'use_max_pool': self.use_max_pool}

    def get_activations(self, x, layer_num=0, pre_activations=True):
        # x is generally B x C x ...
        # For activations this is converted to C x B x ... format.
        if layer_num == 0:
            return torch.transpose(x, 0, 1)
        if pre_activations:
            cur_hook = hook.Hook(self.all_layers[layer_num - 1])
        else:
            raise NotImplementedError
        self.eval()
        _ = self.forward(x)
        cur_hook.close()
        return cur_hook.output.transpose(0, 1).detach()


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


#################### TESTS ########################


def test_resnet18():
    model = resnet18(num_classes=10)
    print(model)
    print(model.all_layers)
    print(model.prev_layers_list)
    print(model.prev_similar_layer)


if __name__ == "__main__":
    test_resnet18()
