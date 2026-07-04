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
from utils import plot_imgs
from dataclasses import dataclass

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

"""画像の描画"""
# sample_data = next(iter(train_loader))
# sample_imgs = sample_data[0]
# sample_labels = sample_data[1]
# print(sample_labels)
# print(sample_data)
# print(math.ceil(4/3), type(math.ceil(4/3)))
# print(type(sample_data))
# print(sample_data[0].shape)

# print(len(sample_imgs))

# feature_imgs = []
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# @dataclass
# class SampleClass:
#     name : str
#     output : torch.Tensor

# for i in range(len(sample_labels)):
#     img, label = sample_imgs[i], sample_labels[i]
#     ctx = SampleClass(classes[label], img)
#     feature_imgs.append(ctx)

# for i in feature_imgs:
#     print(f"name : {i.name}, {i.output.shape}")

# plot_imgs(feature_imgs, PLOT_NUM, "public/img/plot_imgs.png")

"""ptファイルの読み込み"""
data = torch.load(r"C:/Users/siran/ML/cnn_visualization/public/pt/VGG16_train_epoch2_batch1.pt", weights_only=False)
for i in data:
    print(i.layer_name, i.input_shape, i.output_shape)