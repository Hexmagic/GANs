from torch import nn
from util.layers import BottleBlock
import torch
from torch.nn import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),  # batch, 32, 96
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 24
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),  # batch, 64, 
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 12
        )
        self.fc = nn.Sequential(nn.Linear(128 * 12 * 12, 1024),
                                nn.LeakyReLU(0.2, True), nn.Linear(1024, 1),
                                nn.Sigmoid())

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, num_feature=96 * 96 * 3 * 4):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(True))
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 50, 3, stride=1, padding=1),  # batch,50,96*2, 96*2
            nn.BatchNorm2d(50),
            nn.ReLU(True))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 96*2, 96*2
            nn.BatchNorm2d(25),
            nn.ReLU(True))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 2, stride=2),  # batch, 3, 96,96
            nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 96 * 2, 96 * 2)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x
