import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from nfnets import WSConv2d, ScaledStdConv2d
from functools import partial
import torch.nn.functional as F
from .classic_models import get_classic_model

import torch.nn as nn
from collections import OrderedDict


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output




__all__ = ['nf_ResNet', 'nf_resnet18', 'nf_resnet34', 'nf_resnet50', 'nf_resnet101',
           'nf_resnet152', 'nf_resnext50_32x4d', 'nf_resnext101_32x8d',
           'nf_wide_resnet50_2', 'nf_wide_resnet101_2']

_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)

ignore_inplace = ['gelu', 'silu', 'softplus', ]

activation_fn = {
    'identity': lambda x, *args, **kwargs: nn.Identity(*args, **kwargs)(x) * _nonlin_gamma['identity'],
    'celu': lambda x, *args, **kwargs: nn.CELU(*args, **kwargs)(x) * _nonlin_gamma['celu'],
    'elu': lambda x, *args, **kwargs: nn.ELU(*args, **kwargs)(x) * _nonlin_gamma['elu'],
    'gelu': lambda x, *args, **kwargs: nn.GELU(*args, **kwargs)(x) * _nonlin_gamma['gelu'],
    'leaky_relu': lambda x, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)(x) * _nonlin_gamma['leaky_relu'],
    'log_sigmoid': lambda x, *args, **kwargs: nn.LogSigmoid(*args, **kwargs)(x) * _nonlin_gamma['log_sigmoid'],
    'log_softmax': lambda x, *args, **kwargs: nn.LogSoftmax(*args, **kwargs)(x) * _nonlin_gamma['log_softmax'],
    'relu': lambda x, *args, **kwargs: nn.ReLU(*args, **kwargs)(x) * _nonlin_gamma['relu'],
    'relu6': lambda x, *args, **kwargs: nn.ReLU6(*args, **kwargs)(x) * _nonlin_gamma['relu6'],
    'selu': lambda x, *args, **kwargs: nn.SELU(*args, **kwargs)(x) * _nonlin_gamma['selu'],
    'sigmoid': lambda x, *args, **kwargs: nn.Sigmoid(*args, **kwargs)(x) * _nonlin_gamma['sigmoid'],
    'silu': lambda x, *args, **kwargs: nn.SiLU(*args, **kwargs)(x) * _nonlin_gamma['silu'],
    'softplus': lambda x, *args, **kwargs: nn.Softplus(*args, **kwargs)(x) * _nonlin_gamma['softplus'],
    'tanh': lambda x, *args, **kwargs: nn.Tanh(*args, **kwargs)(x) * _nonlin_gamma['tanh'],
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return base_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """1x1 convolution"""
    return base_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, base_conv=base_conv)
        self.activation = activation

        if activation not in ignore_inplace:
            self.act = partial(activation_fn[activation], inplace=True)
        else:
            self.act = partial(activation_fn[activation])
        self.conv2 = conv3x3(planes, planes, base_conv=base_conv)
        self.downsample = downsample
        self.stride = stride
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = activation_fn[self.activation](x=x) * self.beta

        out = self.conv1(out)
        out = self.act(x=out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out *= self.alpha
        out += identity

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, base_conv=base_conv)
        self.conv2 = conv3x3(width, width, stride, groups,
                             dilation, base_conv=base_conv)
        self.conv3 = conv1x1(
            width, planes * self.expansion, base_conv=base_conv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.alpha = alpha
        self.beta = beta
        self.activation = activation
        if activation not in ignore_inplace:
            self.act = partial(activation_fn[activation], inplace=True)
        else:
            self.act = partial(activation_fn[activation])

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = activation_fn[self.activation](x) * self.beta

        out = self.conv1(out)
        out = self.act(x=out)

        out = self.conv2(out)
        out = self.act(x=out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out *= self.alpha
        out += identity

        return out


class NFResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(NFResNet, self).__init__()

        assert activation in activation_fn.keys()

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
        self.conv1 = base_conv(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], alpha=alpha, beta=beta, activation=activation, base_conv=base_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], alpha=alpha, beta=beta,
                                       activation=activation, base_conv=base_conv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], alpha=alpha, beta=beta,
                                       activation=activation, base_conv=base_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], alpha=alpha, beta=beta,
                                       activation=activation, base_conv=base_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='linear')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, alpha: float = 0.2, beta: float = 1.0,
                    activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride, base_conv=base_conv),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, alpha=alpha, beta=beta, activation=activation,
                            base_conv=base_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                alpha=alpha, beta=beta, activation=activation,
                                base_conv=base_conv))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _nf_resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        alpha: float,
        beta: float,
        activation: str,
        base_conv: nn.Conv2d,
        **kwargs: Any
) -> NFResNet:
    model = NFResNet(block, layers, alpha=alpha, beta=beta, activation=activation, base_conv=base_conv, **kwargs)
    return model


def nf_resnet18(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d,
                **kwargs: Any) -> NFResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _nf_resnet('resnet18', BasicBlock, [2, 2, 2, 2], alpha=alpha, beta=beta, activation=activation,
                      base_conv=base_conv,
                      **kwargs)


def nf_resnet34(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d,
                **kwargs: Any) -> NFResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    return _nf_resnet('resnet34', BasicBlock, [3, 4, 6, 3], alpha=alpha, beta=beta, activation=activation,
                      base_conv=base_conv,
                      **kwargs)


def nf_resnet50(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d,
                **kwargs: Any) -> NFResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    return _nf_resnet('resnet50', Bottleneck, [3, 4, 6, 3], alpha=alpha, beta=beta, activation=activation,
                      base_conv=base_conv,
                      **kwargs)


def nf_resnet101(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                 base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    return _nf_resnet('resnet101', Bottleneck, [3, 4, 23, 3], alpha=alpha, beta=beta, activation=activation,
                      base_conv=base_conv,
                      **kwargs)


def nf_resnet152(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                 base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    return _nf_resnet('resnet152', Bottleneck, [3, 8, 36, 3], alpha=alpha, beta=beta, activation=activation,
                      base_conv=base_conv,
                      **kwargs)


def nf_resnext50_32x4d(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                       base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _nf_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                      alpha=alpha, beta=beta, activation=activation, base_conv=base_conv, **kwargs)


def nf_resnext101_32x8d(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                        base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _nf_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                      alpha=alpha, beta=beta, activation=activation, base_conv=base_conv, **kwargs)


def nf_wide_resnet50_2(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                       base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    kwargs['width_per_group'] = 64 * 2
    return _nf_resnet('wide_nf_resnet50_2', Bottleneck, [3, 4, 6, 3],
                      alpha=alpha, beta=beta, activation=activation, base_conv=base_conv, **kwargs)


def nf_wide_resnet101_2(alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu',
                        base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    kwargs['width_per_group'] = 64 * 2
    return _nf_resnet('wide_nf_resnet101_2', Bottleneck, [3, 4, 23, 3],
                      alpha=alpha, beta=beta, activation=activation, base_conv=base_conv, **kwargs)

class FLNet(nn.Module):
    def __init__(self, num_class : int = 10):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

import torch

def get_model(model_name: str = "resnet18"):
    if model_name == "resnet18":
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    elif model_name == "nf_resnet18":
        return nf_resnet18()
    elif model_name == "nf_resnet34":
        return nf_resnet34()
    elif model_name == "FLNet":
        return FLNet()
    elif model_name == "squeezenet1_0":
        return torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
    elif model_name == "AllCNN":
        return get_classic_model("AllCNN")
    elif model_name == "SmallAllCNN":
        return get_classic_model("SmallAllCNN")
    elif model_name == "ResNet18_small":
        return get_classic_model("ResNet18_small")
    elif model_name == "ResNet18_small_test":
        return get_classic_model("ResNet18_small_test")
    elif model_name == "LeNet5":
        return LeNet5()