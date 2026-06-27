import math
from src.layers import Pooling
import torch
from torchvision import transforms, datasets
# print(math.floor(1.2), math.floor(-1.2))
from matplotlib import pyplot as plt
import japanize_matplotlib

seed = 42

torch.manual_seed(seed)

filter_num = 2 # 畳み込み後のチャンネル数
batch_size = 4
filter_size = 3
channels = 0
pool_h = 2
pool_w = 2
stride = 2
pad = 0

# データ読み込み
dataloader  = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transforms.ToTensor(),),
    batch_size=batch_size,
    shuffle=False
)

batch_data = next(iter(dataloader))
imgs = batch_data[0]
labels = batch_data[1]
channels = imgs[0].shape[0]
area = int(math.sqrt(batch_size))

p = Pooling(pool_h, pool_w, stride, pad)

"""プーリング前の可視化
"""
zFig = plt.figure(figsize=(7, 5))
zFig.suptitle("プーリング前")

for i in range(len(imgs)):
    ax = zFig.add_subplot(area, area, i + 1)
    ax.imshow(imgs[i].detach().permute(1, 2, 0))
    ax.axis("off")
    
plt.savefig("プーリング前.png")

"""Forwardの処理
"""

fpool = p.forward(imgs)

fFig = plt.figure(figsize=(7, 5))
fFig.suptitle("プーリング後")

for i in range(len(fpool)):
    ax = fFig.add_subplot(area, area, i + 1)
    ax.imshow(fpool[i].detach().permute(1, 2, 0))
    ax.axis("off")
    
plt.savefig("プーリング後.png")