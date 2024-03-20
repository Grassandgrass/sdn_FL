import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # convolutional layer (sees 1 x14x14 image tensor) (RGB x W x H)
        self.conv1 = nn.Conv2d(1, 96, 3, padding=1, stride=1)
        # convolutional layer (sees 96 x16x16 tensor)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        # convolutional layer (sees 96 x8x8 tensor)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1, stride=1)
        # convolutional layer (sees 192 x4x4 tensor)
        self.conv4 = nn.Conv2d(192, 192, 3, padding=1, stride=1)

        # linear layer (192 * 2 *2 -> 768)
        self.fc1 = nn.Linear(192 * 2 * 2, 500)
        # linear layer (500 -> 256)
        self.fc2 = nn.Linear(500, 256)
        # linear layer (256 -> 2)
        self.drop1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view((x.shape[0], 14, 14))  # torch.Size([512, 14, 14])
        x = x.unsqueeze(1)  # torch.Size([512, 1, 14, 14])
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x)) # torch.Size([512, 192, 2, 2])
        # print(x.shape)

        # flatten image input
        x = x.view(-1, 192 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__=='__main__':
	model = Model()
	a = torch.randn(512,1,14,14)
	abc = model(a)
	print(abc)