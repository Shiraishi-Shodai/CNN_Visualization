import math
from layers import Convolution
import torch
from torchvision import transforms, datasets
# print(math.floor(1.2), math.floor(-1.2))

seed = 42

torch.manual_seed(seed)

filter_num = 2
batch_size = 3
filter_size = 3
channels = 0

dataloader  = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transforms.ToTensor(),),
    batch_size=batch_size,
    shuffle=False
)

batch_data = next(iter(dataloader))
imgs = batch_data[0]
labels = batch_data[1]

channels = imgs[0].shape[0]

w = torch.randn(filter_num, channels, filter_size, filter_size)
b = torch.randn(filter_num)

print(w.shape, imgs.shape)

c = Convolution(w, b, stride=1, pad=0)
col = c.forward(imgs)

print(col.shape)