# senet
from functools import reduce
import torch
from torch import nn, autograd
import torchvision.models as models
from torch.nn import functional as F
import utils


import torch.nn as nn
import math

from utils.fmodule import FModule


class BasicResidualSEBlock(FModule):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(FModule):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class Model(FModule):
    # def __init__(self, block = BasicResidualSEBlock, num_block=[2,2,2,2], class_num=10):
    def __init__(self, block=BottleneckResidualSEBlock, num_block=[3, 4, 6, 3], class_num=2):
        super().__init__()

        self.in_channels = 64

        # self.fc1 = nn.Linear(122, 1024)
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.pre = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, 64, num_block[0],  1)
        self.stage2 = self._make_stage(block, 128, num_block[1], 2)
        self.stage3 = self._make_stage(block, 256, num_block[2], 2)
        self.stage4 = self._make_stage(block, 512, num_block[3], 2)
        # self.stage5 = self._make_stage(block, block_num[4], 512*2, 2)

        # self.linear = nn.Linear(self.in_channels, class_num)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, class_num)

    def forward(self, x):  #torch.Size([512, 196])
        # x = self.fc1(x)  #torch.Size([512, 1024])
        # x = x.view((x.shape[0], 32, 32))  # torch.Size([512, 32, 32])
        # x = x.unsqueeze(1)  # torch.Size([512, 1, 32, 32])
        # output = self.conv0(x)

        output = self.pre(x) # 64*32*32  64*64*64
        output = self.stage1(output)  # 64*32*32 64*64*64
        output = self.stage2(output) # 128*16*16 128*32*32
        output = self.stage3(output)  # 256*8*8 256*16*16
        output = self.stage4(output)  # 512*4*4 512*8*8
        # x = self.stage5(x)

        # output = F.adaptive_avg_pool2d(output, 1)
        # output = x.view(output.size(0), -1)
        # output = self.linear(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def _make_stage(self, block, out_channels, num, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)

def seresnet18():
    return Model(BasicResidualSEBlock, [2, 2, 2, 2])

def seresnet34():
    return Model(BasicResidualSEBlock, [3,4,6,3])

def seresnet50():
    return Model(BottleneckResidualSEBlock, [3, 4, 6, 3])

def seresnet101():
    return Model(BottleneckResidualSEBlock, [3, 4, 23, 3])

def seresnet152():
    return Model(BottleneckResidualSEBlock, [3, 8, 36, 3])



import torch

if __name__=='__main__':
	model = seresnet34().cuda()
	a = torch.randn(128,32,32,32).cuda()
	abc = model(a)
	print(abc)