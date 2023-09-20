from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import MNISTEncoder
import numpy as np
from time import time
from params import base_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:\t", device)

ds = MNIST(root=base_path, download=True, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
dl = DataLoader(ds, batch_size=len(ds)//16, shuffle=True)

# Testing
x,y = next(iter(dl))
x,y = x.to(device), y.to(device)

loss_func = nn.CrossEntropyLoss()
model = MNISTEncoder().to(device)
pred = model(x)
loss = loss_func(pred, y)
print("Initial Loss:\t", loss.item())

model = MNISTEncoder().to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

losses = []

time1 = time()
for epoch in range(epochs):
    print(f"Running Epoch {epoch}")
    losses_epoch = []
    for x, y in dl:
        optim.zero_grad()
        pred = model(x)
        loss = loss_func(pred, y)
        losses_epoch.append(loss.item())
        loss.backward()
        optim.step()
    losses.append(np.mean(losses_epoch))
    print("Loss:\t", losses[-1])
    print("Epoch Run Time:\t", time()-time1, "s")