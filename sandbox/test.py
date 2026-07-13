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
import glob
import os

# train_data = datasets.CIFAR10(
#     root="data/cifar10", 
#     train=True, 
#     download=True, 
#     transform=transforms.ToTensor(),
# )

# # data loaderの作成
# train_loader = torch.utils.data.DataLoader(
#     train_data,
#     batch_size=17,
#     shuffle=True
#     )

# PLOT_NUM = 4

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

# """ptファイルの読み込み"""
# data = torch.load(r"C:/Users/siran/ML/cnn_visualization/public/pt/VGG16_train_epoch2_batch1.pt", weights_only=False)
# for i in data:
#     print(i.layer_name, i.input_shape, i.output_shape)

# a = torch.tensor([2, 0, 2])
# cond = a == 2
# r = cond.nonzero(as_tuple=True)[0][0].item() if len(cond.nonzero(as_tuple=True)[0]) > 0 else None
# # print(type(r.item()), r.item())
# print(r)

# dir_path = r"public/pt/miniVGG"
# file_list = glob.glob(f"{dir_path}/*.pt")
# print(file_list)

# from  matplotlib import pyplot as plt
# sample = next(iter(train_loader))
# sample_x, sample_t = sample[0], sample[1]
# for i in range(len(sample_t)):
#     if sample_t[i] != 2:
#         continue
#     plt.imshow(sample_x[i].permute(1, 2, 0).detach().cpu())
#     print(i)
#     plt.title(classes[sample_t[i]])
#     plt.show()
    
# a = torch.tensor([1, 2, 3])
# print(a.sum().item())

# sum_params = 0
# sum_grads = 0

# for layer in self.model.layers:
#     for param, grad in zip(layer.params, layer.grads):
#         sum_params += param.sum().item()
#         sum_grads += grad.sum().item()
# print(self.current_epoch, sum_params, sum_grads)

# a = torch.arange(10) + 0.1
# a = a.reshape(2, -1)
# # r = calc_l2_norm(a)
# # print(r)

# print(a.norm(p=2).item())

# a = list(range(10))
# print(a)
# print(a[1:7:2])

# n = 1
# c = 1
# hh = 5
# ww = 5

# pixels = n * c * hh * ww
# pad = 1
# img = torch.range(1, pixels).reshape(1, 1, 5, 5)
# N, C, H, W = img.shape
# filter_h = 3
# filter_w = 3
# stride = 1
# out_h = int((H + pad * 2 - filter_h) / stride) + 1
# out_w = int((W + pad * 2 - filter_w) / stride) + 1
# pad_img = torch.zeros(N, C, pad*2 + H, pad*2+W)

# pad_img[:, :, pad:-pad, pad:-pad] = img

# col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))

# # print(pad_img, out_h, out_w)

# for y in range(filter_h):
#     y_max = y + stride * out_h
#     for x in range(filter_w):
#         x_max = x + stride * out_w
#         # 画像を一度の処理で複数の領域から1ピクセルずつ切り出すイメージ
#         # 各畳み込み位置に対応する需要やを1度にまとめて抜き出している
#         col[:, :, y, x, :, :] = pad_img[:, :, y:y_max:stride, x:x_max:stride]
#         print(y, x)
# print(col.shape)
# print(pad_img)
# print(col)

# col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
# print(col)
# b = torch.arange(25).reshape(5, -1)
# print(b)
# print(b[0:5:2, 0:5:2])

# a = torch.arange(10, dtype=torch.float32).reshape(2, -1)
# a[0][3] = 1.1
# print(a)
# # print(torch.where(a > 5, a, 0))
# # print(torch.bincount(a.flatten())[1].item())
# print(torch.count_nonzero(a).item())
# print((len(a.flatten()) - torch.count_nonzero(a).item()) / len(a.flatten()))

bias = torch.arange(10)
db = torch.ones((1, 10))
print(bias)
print(db)

print(bias - db)