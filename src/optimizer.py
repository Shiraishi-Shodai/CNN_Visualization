import torch
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """基底クラス
    """
    def pre_forward(self, params):
        pass
    def pre_update(self, params):
        pass
    
    @abstractmethod
    def update(self, params, grads, weight_decay_targets):
        pass
class SGD(Optimizer):
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

class Momentum(Optimizer):
    """モーメンタム
        
    Parameters
    ----------
    lr : 学習率
    momentum  : 1ステップ前の重みを加える割合
    weight_decay: L2正則化の係数
    """
    def __init__(self,  lr=0.01, momentum=0.5, weight_decay=0.0001):
        self.velocity = None
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
    
    def update(self, params, grads, weight_decay_targets):
        if self.velocity is None:
            self.velocity = [torch.zeros_like(param) for param in params]

        for i in range(len(params)):
            # weight decay
            if weight_decay_targets[i]:
                grads[i] += self.weight_decay * params[i]
                
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]

class NAG(Optimizer):
    def __init__(self, lr=0.01, momentum=0.5, weight_decay=0.0):
        """ネステロフの加速勾配法
        https://cs231n.github.io/neural-networks-3/#sgd
        
        Parameters
        ----------
        lr : 学習率
        momentum  : 1ステップ前の重みを加える割合
        weight_decay: L2正則化の係数
        
        W = 10
        ↓ lookahead
        W = 8.2
        ↓ forward
        Loss
        ↓ backward
        grad = ∇L(8.2)
        ↓ restore
        W = 10
        ↓ update
        W = 10 + v_new
        """
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None # 1ステップ前の重みの更新量
        self.backup_params = None # backward後に戻すための元のパラメータ
        
    def update(self, params, grads, weight_decay_targets):
        """パラメータの更新
        """
        if self.backup_params is not None:
            raise RuntimeError("restoreをupdateの前に呼び出す必要があります。")

        if self.velocity is None:
            self.velocity = [torch.zeros_like(param) for param in params]
            
        for i in range(len(params)):
            # weight decay
            if weight_decay_targets[i]:
                grads[i] += self.weight_decay * params[i]
            
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]
    
    def pre_forward(self, params):
        """先読みしたパラメータに置き換える
        """
        
        if self.backup_params is not None:
            raise ValueError("lookaheadが2回呼び出されました")

        if self.velocity is None:
            self.velocity = [torch.zeros_like(param) for param in params]
            
        self.backup_params = [param.clone() for param in params]

        for i in range(len(params)):
            params[i] += self.momentum * self.velocity[i]
        
    def pre_update(self, params):
        """元のパラメータに置き換える
        """
        if self.backup_params is None:
            raise RuntimeError("restoreがlookaheadの前に呼び出されました。")

        for i in range(len(params)):
            # モデルが各レイヤーのparamを参照することは変えず、値のみを置き換える
            params[i].copy_(self.backup_params[i])
        self.backup_params = None