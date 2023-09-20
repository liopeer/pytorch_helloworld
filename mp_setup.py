import os
from typing import Any
import torch
from torch.distributed import init_process_group, destroy_process_group

class DDP_Proc_Group:
    def __init__(self, function) -> None:
        self.function = function

    def __call__(self, rank: int, *args, **kwargs) -> None:
        self._ddp_setup(rank, kwargs.pop("world_size"))
        self.function(*args, **kwargs)
        destroy_process_group()

    def _ddp_setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)