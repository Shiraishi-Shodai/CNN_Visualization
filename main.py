import math
from layers import Affine
import torch
from utils import softmax, cross_entropy

torch.manual_seed(42)

def main():
    x = torch.arange(10).reshape(2, -1)
    y = softmax(x)
    t = torch.zeros_like(y)
    t[0, 0] = 1
    t[1, 2] = 1

    loss = cross_entropy(y, t)
    
    N, D = x.shape
    
    dx = (y - t) / N
    print(dx)


if __name__ == "__main__":
    main()