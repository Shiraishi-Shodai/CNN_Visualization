import torch
from dataclasses import dataclass

@dataclass
class LayerRecord:
    name : str
    input_tensor: torch.Tensor
    output_tensor: torch.Tensor
    input_shape : tuple
    output_shape : tuple

@dataclass
class ParamRecord:
    name : str
    param_norm : float

@dataclass
class GradientRecord:
    name : str # layer name
    param_name : str
    grad_norm : float
    grad_std : float
    grad_mean : float
    grad_max : float
    grad_min : float
@dataclass
class TrainerMetadata:
    name : str
    mode : str
    epoch : int
    batch : int
    plot_img_idx : int
    
@dataclass
class Record:
    metadata: TrainerMetadata

    layer_records: list[LayerRecord]
    param_records: list[ParamRecord]
    gradient_records: list[GradientRecord]