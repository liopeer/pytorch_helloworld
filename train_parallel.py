from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import MNISTEncoder
import numpy as np
from time import time
from params import base_path
from ddp_trainer import Trainer
from torch.utils.data.distributed import DistributedSampler
from ddp_setup import ddp_setup
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

def load_train_objs():
    train_set = MNIST(root=base_path, download=True, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
    model = MNISTEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

def main(rank: int, world_size: int, total_epochs, save_every, batch_size):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, gpu_id=rank, save_every=save_every, checkpoint_folder=base_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Device Count:", world_size)
    mp.spawn(main, args=(world_size, 3, 10, 3000), nprocs=world_size)