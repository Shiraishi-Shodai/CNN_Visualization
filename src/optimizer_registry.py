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
def build_Momentum(optimizer_config):
    lr = optimizer_config["lr"]
    momentum = optimizer_config["momentum"]
    weight_decay = optimizer_config["weight_decay"]

    return Momentum(lr=lr, momentum=momentum, weight_decay=weight_decay)

@register("NAG")
def build_Momentum(optimizer_config):
    lr = optimizer_config["lr"]
    momentum = optimizer_config["momentum"]
    weight_decay = optimizer_config["weight_decay"]

    return NAG(lr=lr, momentum=momentum, weight_decay=weight_decay)
