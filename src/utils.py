import torch 
import numpy as np
from yaml import safe_load
from pathlib import Path
import math
from matplotlib import pyplot as plt
import japanize_matplotlib
import seaborn as sns

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

def plot_imgsWithLabel(data, correct_labels, pred_labels, num_cols, save_filename, axes_title=True, title="正解と予測の描画"):
    """予測した画像とラベルを描画する
    Parameter
    ---------
    data : images
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
            tensor_img = data[idx]

            display_img = tensor_to_display_image(tensor_img) # 描画用画像を取得
            channel_nums = display_img.shape[0] # cmapを取得
            cmap = get_cmap(channel_nums)
            axes[r, c].imshow(display_img, cmap=cmap)
            axes[r, c].axis("off")
            if axes_title:
                color = "yellowgreen" if correct_labels[idx] == pred_labels[idx] else "red"
                axes[r, c].set_title(f"Correct : {correct_labels[idx]}\nPred : {pred_labels[idx]}", color=color)
    if axes_title:
        fig.tight_layout() # 各axisが重ならないように設定
    fig.suptitle(title)
    plt.savefig(save_filename)


# def view_train_valid_history(train_metrics, valid_metrics, filename="public/img/score_loss.png"):
#     train_scores = [i.accuracy for i in train_metrics]
#     valid_scores = [i.accuracy for i in valid_metrics]
#     train_losses = [i.loss for i in train_metrics]
#     valid_losses = [i.loss for i in valid_metrics]
    
#     assert len(train_scores) == len(valid_scores)
#     assert len(train_losses) == len(valid_losses)
    
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#     epochs = torch.arange(1, len(train_scores) + 1)
#     axes[0].plot(epochs, train_scores, label="train", color="blue")
#     axes[0].plot(epochs, valid_scores, label="valid", color="red")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Score")
#     axes[0].set_title("Accuracy")
#     axes[0].legend()
    
#     axes[1].plot(epochs, train_losses, label="train", color="blue")
#     axes[1].plot(epochs, valid_losses, label="valid", color="red")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Loss")
#     axes[1].set_title("Loss")
#     axes[1].legend()
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close(fig)

def extract_metric(metrics, attr):
    """EpochMetricsからプロパティをリストにして返す
    """
    return [getattr(m, attr) for m in metrics]

def filter_metrics(plots):
    """表示対象の指標のみフィルター
    """
    enabled_plots = []
    for plot_dict in plots:
        if not plot_dict["enabled"]:
            continue
        
        enabled_plots.append(plot_dict)
    
    return enabled_plots

def plot_metrics(ax, epochs, train_value, valid_value, title, ylabel):
    """一つの指標をプロット(train, valid)
    """
    ax.plot(epochs, train_value, label="train")
    ax.plot(epochs, valid_value, label="valid")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend()

def plot_epoch_metrics(train_metrics, valid_metrics, plots, save_filename):
    """Epochごとの指標をプロット
    横軸: Epoch
    縦軸: plotsで与える任意の指標(accuracyなど)
    """
    enabled_plots = filter_metrics(plots)
    fig, axes = plt.subplots(1, len(enabled_plots), figsize=(10, 4))
    epochs = torch.arange(len(train_metrics))
    
    for ax, plot_dict in zip(axes, enabled_plots):
        attr = plot_dict["metric"]
        title = plot_dict["title"]
        ylabel = plot_dict["ylabel"]
        
        train_value = extract_metric(train_metrics, attr)
        valid_value = extract_metric(valid_metrics, attr)
        
        assert len(train_value) == len(valid_value)
        
        plot_metrics(ax, epochs, train_value, valid_value, title, ylabel)
    
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)

def confusion_matrix_calc(confusion_matrix, metrics="total accuracy"):
    """混同行列からaccuracyなどを計算する関数
    """
    match metrics:
        case "total accuracy":
            # float()で小数の割り算を実現。clampで0で割ることを防ぐ(クラスごとのaccuracyの計算にも使える)
            correct = torch.sum(confusion_matrix.diag().float()).item()
            total = torch.sum(confusion_matrix.sum(dim=1).clamp(min=1)).item()
            score = (correct / total) * 100
            return score
        
        case "class accuracy":
            # float()で小数の割り算を実現。clampで0で割ることを防ぐ(クラスごとのaccuracyの計算にも使える)
            correct = confusion_matrix.diag().float()
            total = confusion_matrix.sum(dim=1).clamp(min=1)
            score = (correct / total) * 100
            return score

def view_confusion_matrix(train_confusion_matrix, valid_confusion_matrix, ticks, save_filename):
    """学習・検証のconfusion_matrixを表示
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    cm_dict = {
        "Train": train_confusion_matrix,
        "Valid": valid_confusion_matrix
    }
    
    for ax, (title, cm) in zip(axes, cm_dict.items()):
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=ticks,
            yticklabels=ticks
        )
        ax.set_xlabel("Prediction")
        ax.set_ylabel("True")
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)

def view_class_accuracy(train_class_accuracy, valid_class_accuracy, ticks, save_filename):
    """混同行列からクラスごとの正解率を棒グラフで表示する
    """
    class_accuracy_dict = {
        "Train": train_class_accuracy,
        "Valid": valid_class_accuracy
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (title, class_accuracy) in zip(axes, class_accuracy_dict.items()):
        sns.barplot(
            x=ticks,
            y=class_accuracy,
            ax=ax,
        )
        
        ax.set_xlabel("Accuracy by class")
        ax.set_title(title)
    
    plt.savefig(save_filename)
    plt.close(fig)

def view_label_distribution(train_labels, valid_labels, test_labels, ticks, save_filename):
    """dataloader内の各ラベルの個数を表示する(分布に偏りがないか確かめるため)
    """
    
    label_dict = {
        "Train": train_labels,
        "Valid": valid_labels,
        "Test": test_labels
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (title, labels) in zip(axes, label_dict.items()):
        sns.barplot(
            x=ticks,
            y=labels,
            ax=ax
        )
        
        ax.set_xlabel("class")
        ax.set_ylabel("data num")
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)
    
def get_dataset_labels(dataset):
    """datasetからlabelリストを取得する
       datasetがSubsetでも問題なし
    """
    labels = []

    for _, label in dataset:
        labels.append(label)
    
    return labels
    
    