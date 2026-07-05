import torch
from optimizer import *

OPTIMIZER_BUILDERS = {}

def register(name):
    def decorator(func):
        OPTIMIZER_BUILDERS[name] = func
        return func

    return decorator

def get_optimizer_builder(name):
    if name not in OPTIMIZER_BUILDERS:
        raise ValueError(f"{name}はlayer registerに登録されていません")
    return OPTIMIZER_BUILDERS[name]

@register("SGD")
def build_SGD(params, optimizer_config):
    lr = optimizer_config["lr"]

    return SGD(params=params, lr=lr)

@register("Momentum")
def build_Momentum(params, optimizer_config):
    lr = optimizer_config["lr"]
    mu = optimizer_config["mu"]

    return Momentum(params=params, lr=lr, mu=mu)
