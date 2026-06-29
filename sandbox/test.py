"""損失の計算
"""

    # x = torch.arange(10).reshape(2, -1)
    # y = softmax(x)
    # # t = torch.zeros_like(y)
    # t = torch.tensor([0, 2])
    # # t[0, 0] = 1
    # # t[1, 2] = 1

    # loss = cross_entropy_error(y, t)
    
    # N, D = x.shape
    
    # dx = y.clone()
    # dx[torch.arange(N), t] -= 1
    # dx = dx / N
    
    # print(y)
    # print(dx)
    
import torch

# x = torch.randint(1, 10, (10,)).reshape(2, -1)
# print(x)

# print(x.argmax(dim=1))

x = [1, 2]
b = [[3, 1]]
c = [[[4, 5, 6]]]
x += b
x += c

y = []
g = [[1, 2]]
h = [[[3, 4]]]
y += g
y += h
print(x, y)