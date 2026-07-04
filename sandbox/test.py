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
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import math

train_data = datasets.CIFAR10(
    root="data/cifar10", 
    train=True, 
    download=True, 
    transform=transforms.ToTensor(),
)

# data loaderの作成
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=17,
    shuffle=True
    )

PLOT_NUM = 4

# x = torch.randint(1, 10, (10,)).reshape(2, -1)
# print(x)

# print(x.argmax(dim=1))

# x = [1, 2]
# b = [[3, 1]]
# c = [[[4, 5, 6]]]
# x += b
# x += c

# y = []
# g = [[1, 2]]
# h = [[[3, 4]]]
# y += g
# y += h
# print(x, y)

# x = torch.arange(5)
# y = x.clone()

# y[0] = 100

# print(x)
# print(y)

def plot_data():
    sample_data = next(iter(train_loader))
    sample_imgs = sample_data[0]
    sample_labels = sample_data[1]
    print(sample_labels)
    N, C, H, W = sample_imgs.shape
    print(N, C, H, W)
    rows = math.ceil(N / PLOT_NUM) # 余りも表示せれる行数


    fig, axes = plt.subplots(rows, PLOT_NUM)
    print(axes, type(axes))

    # プロット時は余りのデータまで
    for r in range(rows):
        for c in range(PLOT_NUM):
            idx = r * PLOT_NUM + c
            if idx >= N :
                break
            print(sample_imgs[idx].shape, sample_labels[idx])
            axes[r, c].imshow(sample_imgs[idx].permute(1, 2, 0).detach().numpy())
            axes[r, c].axis("off")

    plt.show()
# print(sample_data)
# print(math.ceil(4/3), type(math.ceil(4/3)))
# print(type(sample_data))
# print(sample_data[0].shape)

