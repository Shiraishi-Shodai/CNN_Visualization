import torch
from layers import Affine
import torch.nn as nn
import numpy as np
from optimizer import SGD

class SimpleDense:
    def __init__(self, neurons):
        self.params = []
        self.grads = []
        self.layers = []

        for i in range(len(neurons) - 1):
            fan_in = neurons[i]
            fan_out = neurons[i + 1]
            
            # Heの初期化
            W = nn.Parameter(torch.tensor(np.random.normal(
                loc=0,
                scale=np.sqrt(2/fan_in),
                size=(fan_in, fan_out)
            ).astype("float32")), requires_grad=False)
            
            b = nn.Parameter(torch.tensor(np.zeros([fan_out]).astype("float32")), requires_grad=False)
            
            layer = Affine(W, b)
            self.layers += [layer]
            self.params += [W, b]
            self.grads += layer.grads
    
    def forward(self, x):
        out = None
        
        for layer in self.layers:
            x = layer.forward(x)
        
        out = x
        return out
    
    def backward(self, dout):
        dx = None
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        dx = dout
        return dx
    
x = torch.randint(1, 100, (10, 2)).to(torch.float32)
sa = SimpleDense((2, 3, 4, 5))

out = sa.forward(x)
print(out)

dout = sa.backward(out)
print(dout)

print(f"params")
for i in sa.params:
    print(i.shape)

print(f"grads")
for i in sa.grads:
    print(i.shape)

optim = SGD(0.01)
optim.update(sa.params, sa.grads)