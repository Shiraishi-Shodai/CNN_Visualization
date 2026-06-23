import torch 
import numpy as np

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
    """
    
    N, C, H, W = input_data.shape
    out_h = (H + pad * 2 - filter_h) // 1 + 1
    out_w = (W + pad * 2 - filter_w) // 1 + 1
    
    img = torch.tensor(np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant'))
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    y_max = 0
    x_max = 0
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
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
    