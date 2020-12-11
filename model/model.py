import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FullyConnectedModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


class CNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Output shape = (W−K+2P) / S]+1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x


class LeNet5(BaseModel):
    # Paper Source: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.fn1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fn2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.fn3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.fn4 = nn.Tanh()
        self.fc3 = nn.Linear(84, num_classes)
        self.fn5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.fn2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fn3(x)
        x = self.fc2(x)
        x = self.fn4(x)
        x = self.fc3(x)
        y = self.fn5(x)
        return y