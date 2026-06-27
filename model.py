import torch
from layers import *

class SimpleCNN:
    def __init__(self):
        
        self.layers = []
        
        
    
    def forward(self, x):
        for layer in self.layers:
            out = layer.forward(x)
        
        return out
    
    def backward(self, dout=1):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout