import torch
from layers import Convolution, MaxPooling
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
class SimpleCNN:
    def __init__(self, kernel, cstride, cpad, pool_h, pool_w, pstride, ppad):
        self.params = []
        self.grads = []
        self.layers = []

        FN, C, FH, FW = kernel
        fan_in = C * FH * FW
        fan_out = FN
        
        # 畳み込みの重み
        cWeight = nn.Parameter(torch.tensor(np.random.normal(
            loc=0, 
            scale=np.sqrt(2/fan_in),
            size=(kernel)
        ).astype("float32")), requires_grad=False)
        
        cBias = nn.Parameter(torch.tensor(np.zeros([fan_out]).astype("float32")), requires_grad=False)
        
        # レイヤーの定義
        conv1 = Convolution(cWeight, cBias, cstride, cpad)
        pool1 = MaxPooling(pool_h, pool_w, pstride, ppad)
        
        self.layers = [conv1, pool1]
        
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : 画像(N, C, H, W)

        Returns
        -------
        dx : 入力画像の勾配
        """
        
        out = None
        
        for layer in self.layers:
            x = layer.forward(x)
        
        out = x
        return out
    
    def backward(self, dout=1):
        dx = None
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        dx = dout
        return dx

# パラメータの定義
batch_size = 4
kernel = (4, 3, 3, 3)
cstride = 1
cpad = 0
pool_h = 2
pool_w = 2
pstride = 2
ppad = 0
graph_area = int(math.sqrt(batch_size))

# データの読み込み
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../../data/cifar10", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

data = next(iter(dataloader))
x, y = data

# 畳み込み前の画像を出力
# beforeFig = plt.figure(figsize=(7, 5))

# for i in range(len(x)):
#     ax = beforeFig.add_subplot(graph_area, graph_area, i + 1)
#     ax.axis("off")
#     ax.imshow(x[i].detach().permute(1, 2, 0).numpy())

# plt.savefig("public/img/simple_cnn_before.png")

sc = SimpleCNN(kernel, cstride, cpad, pool_h, pool_w, pstride, ppad)
out = sc.forward(x)

dout = sc.backward(out)
# print(out.size(), out.shape)
arg_max = torch.argmax(out, dim=1)

# 重み、勾配を計算
for i in range(len(sc.params)):
    print(sc.params[i].shape)
    print(sc.grads[i].shape)
    
# 畳み込み後の画像を出力
# sample_conv_img = out[0]
# beforeFig = plt.figure(figsize=(7, 5))

# for i in range(len(sample_conv_img)):
#     ax = beforeFig.add_subplot(graph_area, graph_area, i + 1)
#     ax.axis("off")
#     ax.imshow(sample_conv_img.detach().permute(1, 2, 0)[i].numpy(), cmap="gray")

# plt.savefig("public/img/simple_cnn_after.png")