import torch
from dataclasses import dataclass

@dataclass
class LayerRecord:
    layer_name : str
    input_tensor: torch.Tensor
    output_tensor: torch.Tensor
    input_shape : tuple
    output_shape : tuple
    
@dataclass
class TrainerMetadata:
    model_name : str
    mode : str
    epoch : int
    batch : int