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