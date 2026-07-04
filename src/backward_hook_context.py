import torch
from dataclasses import dataclass

@dataclass
class BackwardHookContext:
    layer : object
    inputs: torch.Tensor
    outputs: torch.Tensor