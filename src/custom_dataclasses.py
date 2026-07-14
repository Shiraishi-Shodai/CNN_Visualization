import torch
from dataclasses import dataclass, field

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
    metadata: TrainerMetadata | None = None

    layer_records: list[LayerRecord] = field(default_factory=list)
    param_records: list[ParamRecord] = field(default_factory=list)
    gradient_records: list[GradientRecord] = field(default_factory=list)
    
@dataclass
class EpochMetrics:
    """EpochMetrics … エポックごとの時系列データ
    """
    loss: float
    accuracy: float

@dataclass
class EvaluationMetrics:
    """EpochMetrics … エポック終了時の評価データ
    """
    confusion_matrix: torch.Tensor
    class_accuracy: torch.Tensor # クラスごとの正解率

@dataclass
class LastMiniBatch:
    x : torch.Tensor
    t : torch.Tensor
    pred : torch.Tensor