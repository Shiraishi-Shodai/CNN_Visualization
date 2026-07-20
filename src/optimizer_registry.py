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
    weight_decay = optimizer_config["weight_decay"]

    return SGD(params=params, lr=lr, weight_decay=weight_decay)

@register("Momentum")
def build_Momentum(params, optimizer_config):
    lr = optimizer_config["lr"]
    mu = optimizer_config["mu"]
    weight_decay = optimizer_config["weight_decay"]

    return Momentum(params=params, lr=lr, mu=mu, weight_decay=weight_decay)
