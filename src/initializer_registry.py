import torch
import torch.nn as nn
import math

INITIALIZERS = {}

def register(name):
    def decorator(func):
        INITIALIZERS[name] = func
        return func
    
    return decorator


@register("he")
def he_initializer(shape, fan_in, fan_out=None):
    # return nn.Parameter(torch.tensor(np.random.normal(
    #     loc=0,
    #     scale=np.sqrt(2/fan_in),
    #     size=shape
    # ).astype("float32")), requires_grad=False)

    return nn.Parameter(torch.normal(mean=0, std=math.sqrt(2/fan_in), size=shape, dtype=torch.float32), requires_grad=False)   
    
    
@register("xavier")
def xavier_initializer(shape, fan_in, fan_out):
    # return nn.Parameter(torch.tensor(np.random.normal(
    #     loc=0,
    #     scale=np.sqrt(2/(fan_in + fan_out)),
    #     size=shape
    # ).astype("float32")), requires_grad=False)
    
    return nn.Parameter(torch.normal(mean=0, std=math.sqrt(2/ (fan_in + fan_out), size=shape), dtype=torch.float32), requires_grad=False)
    
@register("zeros")
def zeros_initializer(shape, fan_in=None, fan_out=None):
    # return nn.Parameter(torch.tensor(np.zeros(shape).astype("float32")), requires_grad=False)
    return nn.Parameter(torch.zeros(size=shape, dtype=torch.float32), requires_grad=False)

@register("ones")
def ones_initializer(shape, fan_in=None, fan_out=None):
    return nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)

def get_initializer(name):
    if name not in INITIALIZERS:
        raise ValueError(f"{name}は登録されていません")
    
    return INITIALIZERS[name]