import torch
from copy import deepcopy
class SGD:
    """
    確率的勾配降下法
    """
    def __init__(self, params, lr=0.01, weight_decay=0.0001):
        self.lr = lr
        self.weight_decay = weight_decay
        
    def update(self, params, grads, weight_decay_targets):
        for i in range(len(params)):
            # weight decay
            if weight_decay_targets[i]:
                grads[i] += self.weight_decay * params[i]
                
            params[i] -= self.lr * grads[i]

class Momentum:
    """モーメンタム
    """
    def __init__(self, params,  lr=0.01, mu=0.5, weight_decay=0.0001):
        self.old_update_params = [torch.zeros_like(param) for param in params]
        self.mu = mu
        self.lr = lr
        self.weight_decay = weight_decay
    
    def update(self, params, grads, weight_decay_targets):
        for i in range(len(params)):
            # weight decay
            if weight_decay_targets[i]:
                grads[i] += self.weight_decay * params[i]
                
            update_params = self.mu * self.old_update_params[i] - self.lr * grads[i]
            params[i] += update_params
            self.old_update_params[i] = update_params
