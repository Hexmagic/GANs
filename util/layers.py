import torch
from torch.nn import *


class ConvBlock(Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 norm=True):
        '''
        参数:
            in_channel ： 输入通道
            out_channel:  输出通道数
            kernel_size:  卷积核大小
            stride: 步长默认1

        '''
        super(ConvBlock, self).__init__()
        layer = [
            Conv2d(in_channel,
                   out_channel,
                   kernel_size,
                   padding=padding,
                   stride=stride,
                   dilation=dilation)
        ]
        if norm:
            layer.append(BatchNorm2d(out_channel))
        self.net = Sequential(*layer)

    def forward(self, X):
        return self.net(X)


class BottleBlock(Module):
    expansion = 4  # 最终输出是输入的几倍

    def __init__(self,
                 channels,
                 norm=True,
                 group=32,
                 stride=1,
                 dilation=1,
                 mode='up'):
        super(BottleBlock, self).__init__()
        '''
        参数:
            channel ： 存储用到的输入和输出通道
            norm:  是否使用norm
        '''
        [first, second, third] = channels
        layers = [
            ConvBlock(first, second, 1, norm=norm, padding=0),
            ReLU(),
            ConvTranspose2d(second,
                            second,
                            3,
                            output_padding=1,
                            padding=1,
                            stride=stride,
                            dilation=dilation)
            if mode == 'up' else ConvBlock(second,
                                           second,
                                           3,
                                           padding=dilation,
                                           stride=stride,
                                           norm=norm,
                                           dilation=dilation),
            ReLU(),
            ConvBlock(second, third, 1, norm=norm, padding=0),
        ]
        self.net = Sequential(*layers)
        self.downsample = Sequential()
        if stride != 1 or first != third:
            self.downsample = ConvTranspose2d(
                first, third, kernel_size=1, output_padding=1,
                stride=stride) if mode == 'up' else Conv2d(first, third,
                                    kernel_size=1, stride=stride)

    def forward(self, X):
        I = X
        if self.downsample:
            I = self.downsample(X)
        X = ReLU(True)(self.net(X) + I)
        return X
