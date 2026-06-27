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
def he_initializer(fan_in, fan_out, shape):
    return nn.Parameter(torch.tensor(np.random.normal(
        loc=0,
        scale=np.sqrt(2/fan_in),
        size=shape
    ).astype("float")), requires_grad=False)
    

@register("xavier")
def he_initializer(fan_in, fan_out, shape):
    return nn.Parameter(torch.tensor(np.random.normal(
        loc=0,
        scale=np.sqrt(2/(fan_in + fan_out)),
        size=shape
    ).astype("float")), requires_grad=False)
    