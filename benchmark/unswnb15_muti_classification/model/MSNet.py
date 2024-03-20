from time import sleep

import torch
from torch import nn
import torch.nn.functional as F
# import torchvision.models as models
import benchmark.unswnb15_muti_classification.model.tools.resnet as models

from utils.fmodule import FModule


class Model(FModule):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, num_classes=10, stride =1):
        super(Model, self).__init__()
        print('------ MSNet model-----')

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # self.fc0 = nn.Linear(196, 1024)

        ## point
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm2d(32),
        )

        ## frame
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
        )

        # resnet = models.resnet18(pretrained=True)
        resnet = models.Model()

        modules = list(resnet.children())[1:-1]  # delete the last fc layer.
        # print(len(modules))
        # print(modules)

        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, num_classes)

        # initial weight
        # initial_model_weight(layers=list(self.children()))
        # print('weight initial finished!')

    def forward(self, x_3d):
        # ResNet CNN
        # print(x_3d.shape)  # torch.Size([8, 196])
        # x_3d = self.fc0(x_3d)  # torch.Size([512, 1024])
        x_3d = x_3d.view((x_3d.shape[0], 14, 14))  # torch.Size([512, 32, 32])
        x_3d = x_3d.unsqueeze(1)

        x_1 = self.conv1_1(x_3d) # torch.Size([8, 1, 32, 32])
        # x_1 = self.conv1_1(self.W * x_3d + x_3d)
        x_2 = self.conv1_2(x_3d) # torch.Size([8, 32, 32, 32])

        # print(x_1.shape, x_2.shape)
        x = self.resnet(torch.cat((x_1, x_2), dim=1)) #([8, 512, 1, 1])

        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training) # torch.Size([8, 512])

        x = self.fc3(x)
        return x

