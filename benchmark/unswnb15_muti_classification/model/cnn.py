from time import sleep

from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # print("forward",x.shape)
        x = self.get_embedding(x)  # torch.Size([512, 128])
        x = self.fc2(x)      # torch.Size([512, 2])
        return x

    def get_embedding(self, x):
        x = x.view((x.shape[0], 14, 14)) # torch.Size([512, 14, 14])
        x = x.unsqueeze(1) # torch.Size([512, 1, 14, 14])
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  #torch.Size([512, 32, 7, 7])
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # torch.Size([512, 64, 3, 3])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])  # torch.Size([512, 576])
        x = F.softmax(self.fc1(x), dim=-1)
        # x = F.relu(self.fc1(x))
        return x

