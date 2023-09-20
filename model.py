import torch.nn as nn
import torch

class MNISTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [2**i for i in range(5)]
        self.encoder = []
        for i in range(4):
            self.encoder.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding="same"))
            self.encoder.append(nn.BatchNorm2d(channels[i+1]))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*self.encoder)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.squeeze())