import torch 
import numpy as np
from yaml import safe_load
from pathlib import Path
import math
from matplotlib import pyplot as plt

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
    N = len(pred)
    score = (pred == t).sum() / N
    
    return score.detach().item()

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    N = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(N), t] + 1e-7)) / N  



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
    # outのサイズ分カーネルは平行移動し、畳み込みを行う
    out_h = (H + pad * 2 - filter_h) // stride + 1
    out_w = (W + pad * 2 - filter_w) // stride + 1
    
    # ゼロパディング
    # img = torch.tensor(np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant'))
    if pad == 0:
        img = input_data
    elif pad > 0:
        img = torch.zeros(N, C, H + pad * 2, W + pad * 2, device=input_data.device, dtype=input_data.dtype)
        img[:, :, pad : -pad , pad : -pad] = input_data
    else:
        raise ValueError("負の値がpaddingとして与えられました。padding : {pad}")

    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w), device=input_data.device, dtype=input_data.dtype)
    y_max = 0
    x_max = 0
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # カーネルの各ピクセルが掛け算する入力ピクセルを、カーネルのピクセルごとに切り出している
            # 例) カーネルの左上を掛け算する入力ピクセルを一度に抜き出す
            # 受容野 : カーネルを適用する領域
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    #  (batch_size, FH, FW, C, out_h, out_w) → (batch_size * out_h * out_w, C * FH * FW)
    # 各行が1つの受容野になるように並べる
    # 各列に受容野内でカーネルの各係数にかける入力画素を並べる
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
    # (勾配受容野, カーネルの各係数) → (N, C, カーネルh, カーネルw, 勾配受容野h, 勾配受容野w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).permute(0, 3, 4, 5, 1, 2)

    # img = torch.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), device=col.device, dtype=col.dtype)
    img = torch.zeros((N, C, H + 2 * pad, W + 2 * pad), device=col.device, dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]

def tensor_to_display_image(tensor_img):
    """tensor画像をplot描画用画像に変換する
    Parameter
    ---------
    tensor_img : (C, H, W)
    """
    C = tensor_img.shape[0]
    display_img = tensor_img.permute(1, 2, 0).numpy()
    
    match C:
        case 3:
            pass
        case 1:
            display_img = display_img.squeeze(2)
        case _:
            display_img = display_img.mean(axis=2)
        
    return display_img

def get_cmap(channel_nums):
    """plt.imshowに使用するcmapを取得
    """
    
    if channel_nums == 3:
        return None
    else:
        return "gray"
    
def plot_imgs(data, num_cols, save_filename, axes_title=True, title=None):
    """取得した画像分描画する
    Parameter
    ---------
    imgs : list[dataclass] dataclass(name, output), output(N, C, H, W)
    num_cols : 横方向に描画する画像の枚数
    save_filename : 保存するファイル名
    """
    
    display_img = None
    cmap = None
    channel_nums = 1
    N = len(data) # 描画する画像の枚数
    rows = math.ceil(N / num_cols) # 余りも含めて表示できる行数
    fig, axes = plt.subplots(rows, num_cols)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # プロット時は余りのデータまで
    for r in range(rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            if idx >= N :
                fig.delaxes(axes[r, c]) # 描画する画像の枚数よりも大きなidxを持つaxは一つずつ削除
                continue
            tensor_img = data[idx].output_tensor[-1]

            display_img = tensor_to_display_image(tensor_img) # 描画用画像を取得
            channel_nums = display_img.shape[0] # cmapを取得
            cmap = get_cmap(channel_nums)
            axes[r, c].imshow(display_img, cmap=cmap)
            axes[r, c].axis("off")
            if axes_title:
                axes[r, c].set_title(data[idx].name)
    if axes_title:
        fig.tight_layout() # 各axisが重ならないように設定
    fig.suptitle(title)
    plt.savefig(save_filename)

# def calc_l2_norm(grads: torch.tensor):
#     l2_norm = 0
    
#     if grads.dim != 1:
#         grads = grads.flatten()
#     l2_norm = torch.sqrt(sum([grad.item() ** 2 for grad in grads]))
    
#     return l2_norm