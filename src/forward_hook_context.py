import torch
from dataclasses import dataclass

@dataclass
class ForwardHookContext:
    layer : object
    inputs: torch.Tensor
    outputs: torch.Tensor