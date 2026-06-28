import torch
from utils import im2col, col2im, softmax, cross_entropy_error

class Layer:
    
    def __init__(self):
        self.params = []
        self.grads = []

    def to(self, device):
        for i in range(len(self.params)):
            self.params[i] = self.params[i].clone().to(device)
            self.grads[i] = self.grads[i].clone().to(device)
    
class ReLU(Layer):
    
    def __init__(self):
        self.x = None
        self.params = []
        self.grads = []
    
    def forward(self, x):
        """
        """
        self.x = x
        return torch.where(self.x > 0, self.x, 0)
    
    def backward(self, dout=1):
        return torch.where(self.x > 0, dout, 0)

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [torch.zeros_like(W), torch.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = x @ W + b
        self.x = x
        
        return out

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        db = torch.sum(dout, dim=0, keepdim=True)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, t, x):
        """
        Parameters
        ----------
        x : 実数を持つ行列(全結合の出力結果)
        t : one-hotベクトルの多値分類結果
            
        Returns
        -------
        loss : 損失(スカラ)
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss.item()
    
    def backward(self, dout=1):
        """
        Parameters
        ----------
        dout : 1
        
        
        Returns
        -------
        dx : softmaxへの入力xの勾配
             y - tを行うことでdxは0か負の実数値になる。
             勾配降下法では、new_w = -now_w * η * (dL / dW)となる
             dWが正の値であれば、new_wはwよりも小さくなる(減少する) → そのニューロンの出力を小さくしようとする
             dWが負の値であれば、new_wはwよりも大きくなる(増加する) → そのニューロンの出力を大きくしようとする
             
             正解ラベルの出力値を予測するニューロンの重みを大きくしたいから、y - tで正解ラベルの勾配dxが0か負の実数値になるようにする。
             そうすることで、dx = 0なら完璧な予測だから更新なし。負の値なら次回予測時に正解ラベルの予測値を大きくしにかかる
        """
        
        N, D = dout.shape
        
        if self.t.size == self.y.size:
            # t : one-hotベクトルの行列想定
            dx = (self.y - self.t) / N
        else:
            # t : ラベルのベクトル想定
            dx = self.y.copy()
            dx[torch.arange(N), self.t] -= 1
            dx = dx / N
        
        return dx
        
class Convolution:
    def __init__(self, weight, bias, stride=1, pad=0):
        self.params = [weight, bias]
        self.grads = [torch.zeros_like(weight), torch.zeros_like(bias)]
        self.stride = stride
        self.pad = pad
        
        # 逆伝搬用データ
        self.x = None
        self.col = None
        self.col_W = None
        
        # 重み
        self.dW = None
        self.db = None
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        
        Returns
        -------
        out : 4次元配列(N, out_h, out_w, 1)
        """
        weight, bias = self.params
        FN, C, FH, FW = weight.shape
        N, C, H, W = x.shape
        out_h = (H + self.pad * 2 - FH) // self.stride + 1
        out_w = (W + self.pad * 2 - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)             # (batch_size * out_h * out_w, C * FH * FW)
        col_W = weight.reshape(FN, -1).T                           #(FN, C * FH * FW) → (C * FH * FW, FN)
        print(col.device, col_W.device, bias.device)
        out = col @ col_W + bias                                 # (batch_size * out_h * out_w, C * FH * FW) @ (C * FH * FW, FN) = (batch_size * out_h * out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2) # (batch_size * out_h * out_w, FN) → (batch_size, FN, out_h, out_w)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    def backward(self, dout):
        weight, bias = self.params
        FN, C, FH, FW = weight.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)
        
        db = torch.sum(dout, dim=0)
        dW = self.col.T @ dout
        dW = dW.permute(1, 0).reshape(FN, C, FH, FW)

        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx
    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.params = []
        self.grads = []
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # (batch_size * out_h * out_w, C * pool_h * pool_w)
        col = col.reshape(-1, self.pool_h*self.pool_w) # (batch_size * out_h * out_w * C, pool_h * pool_w)

        arg_max = torch.argmax(col, dim=1) # プーリング領域での最大値をのインデックスをとる
        out = torch.max(col, dim=1).values # プーリング領域の最大値を求める
        out = out.reshape(N, out_h, out_w, C).permute(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        
        return out
        
    def backward(self, dout):
        dout = dout.permute(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = torch.zeros((dout.numel(), pool_size), device=dout.device, dtype=dout.dtype)
        dmax[torch.arange(self.arg_max.numel()), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
    
class Flatten:
    def __init__(self):
        self.params = []
        self.grads = []
        self.original_shape = None
    
    def forward(self, x):
        """フラットにする
        Parameter
        ---------
        x : (N, C, H, W)
        
        Returns
        -------
        out : (N, C * H * W)
        """
        
        self.original_shape = x.shape
        return x.reshape(self.original_shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.original_shape)
        

class Dropout:
    def __init__(self, dropout_rate):
        self.params = []
        self.grads = []
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, x, mode="train"):
        
        if mode == "train":
            self.mask = self.dropout_rate > torch.rand_like(x)
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)
    
    def backward(self, dout):
        return dout * self.mask
    