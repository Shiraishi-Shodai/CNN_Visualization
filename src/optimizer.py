import torch
from copy import deepcopy
class SGD:
    """
    確率的勾配降下法
    """
    def __init__(self, params, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum:
    """モーメンタム
    """
    def __init__(self, params,  lr=0.01, mu=0.5):
        self.old_update_params = [torch.zeros_like(param) for param in params]
        self.mu = mu
        self.lr = lr
    
    def update(self, params, grads):
        
        for i in range(len(params)):
            update_params = self.mu * self.old_update_params[i] - self.lr * grads[i]
            params[i] += update_params
            self.old_update_params[i] = update_params
