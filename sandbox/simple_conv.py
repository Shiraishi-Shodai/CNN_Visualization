import math
from layers import Convolution
import torch
from torchvision import transforms, datasets
# print(math.floor(1.2), math.floor(-1.2))
from matplotlib import pyplot as plt
import japanize_matplotlib
from utils import col2im

seed = 42

torch.manual_seed(seed)

filter_num = 16 # 畳み込み後のチャンネル数
batch_size = 4
filter_size = 3
channels = 0

# データ読み込み
dataloader  = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transforms.ToTensor(),),
    batch_size=batch_size,
    shuffle=False
)

batch_data = next(iter(dataloader))
imgs = batch_data[0]
input_shape = imgs.shape
labels = batch_data[1]

channels = imgs[0].shape[0]

w = torch.randn(filter_num, channels, filter_size, filter_size)
b = torch.randn(filter_num)

c = Convolution(w, b, stride=1, pad=0)

area = int(math.sqrt(batch_size))
channel_area = int(math.sqrt(filter_num))

"""畳み込み前の可視化
"""

# 畳み込み前の状態を可視化
zFig = plt.figure(figsize=(7, 5))
zFig.suptitle("畳み込み前")

for i in range(len(imgs)):

    ax = zFig.add_subplot(area, area, i + 1)
    ax.axis("off")
    ax.imshow(imgs[i].detach().permute(1, 2, 0))

plt.savefig("畳み込み前.png")

"""Forwardの処理
"""
col = c.forward(imgs)
conv_img1 = col[0]
# 畳み込み前の状態を可視化
fFig = plt.figure(figsize=(7, 5))
fFig.suptitle("1枚目の畳み込み後")

for i in range(len(conv_img1)):
    ax = fFig.add_subplot(channel_area, channel_area, i + 1)
    ax.axis("off")
    ax.imshow(conv_img1[i].detach(), cmap="gray")

plt.savefig("畳み込み後.png")


"""Backwardの処理
畳み込み時に失われたデータがあるため、畳み込み前と同じデータは出力されない
"""
reimgs = c.backward(col)
reimgs = reimgs % 255.0 # 値の範囲を0.0 ~ 255.0にする
reimgs = reimgs.to(torch.long)

# print(torch.max(reimgs), torch.min(reimgs))
# bFig = plt.figure(figsize=(7, 5))
# bFig.suptitle("畳み込み後")

# for i in range(len(reimgs)):
    
#     ax = bFig.add_subplot(area, area, i + 1)
#     ax.imshow(reimgs[i].detach().permute(1, 2, 0))

# plt.savefig("逆畳み込み後.png")