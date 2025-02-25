"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
from time import sleep

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import transforms

from utils.fmodule import FModule
import benchmark.unswnb15_muti_classification.model.resnet18 as models

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Model(FModule):
    def __init__(self, block=BasicBlock, num_block=[16,32], num_classes=10):
        super().__init__()
        #  self.in_channels = 64  Given groups=1, weight of size [64, 3, 3, 3], expected input[512, 1, 8, 8] to have 3 channels, but got 1 channels instead
        self.in_channels = 64
        # self.fc1 = nn.Linear(196, 1024)
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv1_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv2_x = self._make_layer(block, 64, num_block[1], 2)

        self.resnet = models.Model()

        self.conv3_x = self._make_layer(block, 256, num_block[2], (2,3))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(resnet.fc.in_features, 128)
        self.fc1 = nn.Linear(128, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x): # torch.Size([512, 64])
        # x = self.fc1(x)
        x = x.view((x.shape[0], 14, 14))  # torch.Size([512, 32, 32])
        x = x.unsqueeze(1)  # torch.Size([512, 1, 8, 8])
        # print(x.shape)

        # output = self.conv0(x)
        output = self.conv1(x) #torch.Size([512, 64, 8, 8])
        output = nn.MaxPool2d(output, kernel_size=3, stride=2, padding=1)
        output = self.conv2_x(output)
        output = nn.MaxPool2d(output, kernel_size=3, stride=(1,2), padding=1)

        output = self.resnet(output)
        print(output.shape)
        output = self.conv3_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc0(output)
        output = F.dropout(output, p=0.8, training=self.training)
        output = self.fc1(output)
        return output

