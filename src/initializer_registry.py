import numpy as np
import torch
import torch.nn as nn

INITIALIZERS = {}

def register(name):
    def decorator(func):
        INITIALIZERS[name] = func
        return func
    
    return decorator


@register("he")
def he_initializer(shape, fan_in, fan_out=None):
    return nn.Parameter(torch.tensor(np.random.normal(
        loc=0,
        scale=np.sqrt(2/fan_in),
        size=shape
    ).astype("float32")), requires_grad=False)
    
@register("xavier")
def xavier_initializer(shape, fan_in, fan_out):
    return nn.Parameter(torch.tensor(np.random.normal(
        loc=0,
        scale=np.sqrt(2/(fan_in + fan_out)),
        size=shape
    ).astype("float32")), requires_grad=False)
    
@register("zeros")
def zeros_initializer(shape, fan_in=None, fan_out=None):
    return nn.Parameter(torch.tensor(np.zeros(shape).astype("float32")), requires_grad=False)

def get_initializer(name):
    if name not in INITIALIZERS:
        raise ValueError(f"{name}は登録されていません")
    
    return INITIALIZERS[name]