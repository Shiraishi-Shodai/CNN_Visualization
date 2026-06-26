import torch
from utils import *

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
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
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + self.pad * 2 - FH) // self.stride + 1
        out_w = (W + self.pad * 2 - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)             # (batch_size * out_h * out_w, C * FH * FW)
        col_W = self.W.reshape(FN, -1).T                           #(FN, C * FH * FW) → (C * FH * FW, FN)

        out = col @ col_W + self.b                                 # (batch_size * out_h * out_w, C * FH * FW) @ (C * FH * FW, FN) = (batch_size * out_h * out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2) # (batch_size * out_h * out_w, FN) → (batch_size, FN, out_h, out_w)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)
        
        self.db = torch.sum(dout, dim=0)
        self.dW = self.col.T @ dout
        self.dW = self.dW.permute(1, 0).reshape(FN, C, FH, FW)

        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
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
        dmax = torch.zeros((dout.size, pool_size))
        dmax[torch.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape, (pool_size, ))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx