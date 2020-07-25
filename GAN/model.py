from torch import nn
from util.layers import BottleBlock
import torch
from torch.nn import *


class Discriminator(nn.Module):
    '''
    判别器
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = BottleBlock([3, 64, 64], stride=2, mode='down')  # 48*48
        self.down2 = BottleBlock([64, 64, 256], stride=2, mode='down')  #24*24
        self.down3 = BottleBlock([256, 64, 512], stride=2, mode='down')  #12*12
        self.down4 = BottleBlock([512, 128, 1024], stride=2,
                                 mode='down')  #12x12
        self.fc = nn.Sequential(nn.Linear(1024 * 6 * 6, 1024),
                                nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
                                nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_param=49, n_feature=24 * 24 * 3):
        '''
        输入n_param维度的tensor,输出图像size的tensor
        '''
        super(Generator, self).__init__()
        self.fc = nn.Linear(n_param, n_feature)
        self.up1 = BottleBlock([3, 6, 6], stride=2)
        self.up2 = BottleBlock([6, 6, 3], stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 24, 24)
        x = self.up1(x)
        x = self.up2(x)
        return x

