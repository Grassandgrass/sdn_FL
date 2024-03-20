from time import sleep

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import transforms

from utils.fmodule import FModule
import benchmark.unswnb15_muti_classification.model.resnet18 as models
class Model(FModule):
    def __init__(self,  num_classes=10, stride=1):
        super().__init__()
        #  self.in_channels = 64  Given groups=1, weight of size [64, 3, 3, 3], expected input[512, 1, 8, 8] to have 3 channels, but got 1 channels instead
        self.in_channels = 64
        # self.fc1 = nn.Linear(196, 1024)
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.conv1_x = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv2_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=6, stride=stride, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv3_x = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.conv4_x = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        # self.fc = nn.Linear(512//4, num_classes)

    def forward(self, x): # torch.Size([512, 64])
        # x = self.fc1(x)
        x = x.view((x.shape[0], 14, 14))  # torch.Size([512, 32, 32])
        x = x.unsqueeze(1)  # torch.Size([512, 1, 8, 8])

        # output = self.conv0(x)
        output = self.conv1_x(x) # output1 torch.Size([512, 64, 8, 8]
        output = self.conv2_x(output) # output1 torch.Size([512, 128, 5, 5])
        output = self.conv3_x(output)
        output = self.conv4_x(output) #torch.Size([512, 256, 2, 2])

        # output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # output = F.dropout(output, p=0.3, training=self.training)  # torch.Size([512, 256])

        output = self.fc(output)
        # print("output1", output.shape)
        return output

