import torch 
import numpy as np
from yaml import safe_load
from pathlib import Path

def load_yaml(yaml_file : Path):
    """yamlの読み込み
    Parameters
    ----------
    yaml_file : Path
    """
    
    with open(yaml_file, mode="r", encoding="utf-8") as f:
        config = safe_load(f)
        return config

def softmax(x):
    """
    Parameters
    ----------
    x : 全結合での出力結果(N, D)
    
    Returns
    -------
    z : softmax適用後行列
    """
    x_max = x.max(dim=1, keepdim=True).values
    x -= x_max
    exp_x = torch.exp(x)
    z = exp_x / exp_x.sum(dim=1, keepdim=True)
    
    return z

# def cross_entropy_error(y, t):
#     """
#     Parameters
#     ----------
#     y : 出力関数適用後行列 (N, D)
#     t : 正解ラベル(one-hot想定)
    
#     Returns
#     -------
#     loss : バッチ平均をしたスカラ
#     """
#     N, D = y.shape
#     loss = - torch.log(y) * t
#     loss = loss.sum() * (1 / N)
    
#     return loss

def accuracy(pred, t):
    """
    Parameters
    ----------
    pred : 予測値のラベルベクトル(1次元)
    t    : 正解ラベル(1次元)
    """
    pred = torch.tensor([0, 1, 2])
    t = torch.tensor([1, 0, 2])
    N = len(pred)
    score = (pred == t).sum() / N
    
    return score

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    N = y.shape[0]
    return -torch.sum(np.log(y[np.arange(N), t] + 1e-7)) / N  



def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    
    Returns
    -------
    col : 2次元配列(畳み込み後の特徴マップの要素数, カーネルサイズ(filter_h * filter_w))
    
    input_data(batch_size, C, H, W) → col(batch_size * out_h * out_w, C * FH * FW)
    """
    
    N, C, H, W = input_data.shape
    out_h = (H + pad * 2 - filter_h) // stride + 1
    out_w = (W + pad * 2 - filter_w) // stride + 1
    
    img = torch.tensor(np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant'))
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    y_max = 0
    x_max = 0
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    #  (batch_size, FH, FW, C, out_h, out_w) → (batch_size * out_h * out_w, C * FH * FW)
    col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col : 2次元配列(畳み込み後の特徴マップの要素数, カーネルサイズ(filter_h * filter_w))
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
    col : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    """  
    
    N, C, H, W = input_shape
    out_h = (H + pad * 2 - filter_h) // stride + 1
    out_w = (H + pad * 2 - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).permute(0, 3, 4, 5, 1, 2)

    img = torch.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]
    