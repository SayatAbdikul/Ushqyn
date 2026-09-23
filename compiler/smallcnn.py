"""The pinned on-chip MNIST SmallCNN topology used by training and board export."""
import torch.nn as nn
from accelerator_config import AcceleratorConfig


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4,kernel_size=3,stride=1,padding=0,bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(4,8,kernel_size=3,stride=1,padding=0,bias=True)
        self.fc = nn.Linear(8*5*5,AcceleratorConfig.OUT_N,bias=True)

    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        return self.fc(x.reshape(x.size(0),-1))
