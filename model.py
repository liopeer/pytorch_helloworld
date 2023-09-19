import torch.nn as nn
import torch

class MNISTEncoder(nn.Module):
    def __init__(self, kernel_size=3):
        channels = [2**i for i in range(5)]
        self.conv = [nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, padding="same") for i in range(4)]
        self.conv = nn.Sequential(*self.conv)
        self.fc = nn.Linear(7, 10)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.squeeze())