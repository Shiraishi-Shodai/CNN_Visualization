from PIL import Image
import torch
from torchvision import transforms, datasets

### 2次元のフィルターを実装する(N=1, C=1, padding=1, stride=1)
def kernel2D(kernel_shape : tuple, image : torch.tensor, stride:int=1 , padding:int=0):
    """
    stride=1
    padding=0
    
    Args:
        kernel_shape : (C=1, H, W)
        image.shape : (C=1, H, W)
    """
    
    kernel = torch.randint(0, 10, (kernel_shape))
    
    fc, fH, fW = kernel_shape
    C, H, W = image.shape
    
    # フィルターが縦に動く回数
    row_n = H - fH + 1
    # フィルターが横に動く回数
    col_n = W - fW + 1
    
    # 畳み込み画像
    conv_image = torch.zeros((C, row_n, col_n))
        
    for c in range(C):
        for r in range(row_n):
            for w in range(col_n):
                clip_image = image[c, r : r +  fH, w : w + fW]
                scalar = clip_image.flatten() @ kernel.flatten()
                print(f"========channel: {c}, row : {r}, width : {w}===========")
                print(f"切り取り画像 \n {clip_image}")
                print(f"kernel : \n {kernel}")
                print(f"計算結果 \n {scalar}")
                conv_image[c, r, w] = scalar
        
    print(conv_image)
    

### 2次元のフィルターを実装する(N=1, C=1, padding=可変, stride=1)
def kernel2DP(kernel_shape : tuple, image : torch.tensor, padding:int=0, stride:int=1):
    """
    stride=1
    padding=X
    
    Args:
        kernel_shape : (C=1, H, W)
        image.shape : (C=1, H, W)
    """
    
    kernel = torch.randint(0, 10, (kernel_shape), dtype=torch.float)
    
    # パディング
    image = zeroPadding(image, padding)
    print(image)
    
    fc, fH, fW = kernel_shape
    C, H, W = image.shape
    
    # フィルターが縦に動く回数
    row_n = H - fH + 1
    # フィルターが横に動く回数
    col_n = W - fW + 1
    
    # 畳み込み画像
    conv_image = torch.zeros((C, row_n, col_n))

    for c in range(C):
        for r in range(row_n):
            for w in range(col_n):
                clip_image = image[c, r : r +  fH, w : w + fW]
                scalar = clip_image.flatten() @ kernel.flatten()
                print(f"========channel: {c}, row : {r}, width : {w}===========")
                print(f"切り取り画像 \n {clip_image}")
                print(f"kernel : \n {kernel}")
                print(f"計算結果 \n {scalar}")
                conv_image[c, r, w] = scalar
        
    print(conv_image)

def zeroPadding(image, padding=0):
    """
    Args:
        kernel_shape : (N, C, H, W)
    """
    N, C, H, W = image.shape
    
    # paddingが0以下なら処理を終了する
    if padding <= 0 :
        return image
    
    padding_image = torch.zeros(N, C, H + padding * 2, W + padding * 2, dtype=torch.float)
    
    padding_image[:, :, padding : -padding , padding : -padding] = image
    
    return padding_image


def convN(x, kernel, padding, stride):
    """カーネルをずらし、加重和を計算する回数を計算(カーネルで覆えない部分は除外)
    出力画像サイズを返すことと同義
    """
    cn = (((x + padding * 2) - kernel) / stride) + 1
    return int(cn) # カーネル外を除外

### 2次元のフィルターを実装する(N=1, C=1, padding=可変, stride=可変)
def kernel2DPS(kernel_shape : tuple, image : torch.tensor, padding:int=1, stride:int=1):
    """
    Args:
        kernel_shape : (out_channels, input_channels, H, W)
        image.shape : (N, input_channels, H, W)
        padding :
        stride : 
    """
    kernel = torch.randint(0, 10, (kernel_shape), dtype=torch.float)
    
    N, C, H, W = image.shape
    
    out_channels, input_channels, fH, fW = kernel_shape
    
    # フィルターが縦に動く回数
    row_n = convN(H, fH, padding, stride)
    # フィルターが横に動く回数
    col_n = convN(W, fW, padding, stride)
    
    # 1時畳み込み行列(input_kernel分の行列を保持)
    tmp_image = torch.zeros((N, C, row_n, col_n))
    
    # 畳み込み結果行列
    conv_image = torch.zeros((N, out_channels, row_n, col_n))
    
    # パディング
    image = zeroPadding(image, padding)
    print(f"padding image \n {image}")
    print(f"kernel : \n {kernel}")
    
    # outputチャネル数
    for o in range(out_channels):
        # バッチサイズ数
        for n in range(N):
            # インプットチャネル数
            for c in range(C):
                # 画像の高さ
                for r in range(row_n):
                    # 画像の幅
                    for w in range(col_n):
                        clip_image = image[n, c, r * stride : r * stride +  fH, w * stride : w * stride + fW]
                        ckernel = kernel[o, c]
                        scalar = clip_image.flatten() @ ckernel.flatten()
                        print(f"========channel: {c}, row : {r}, width : {w}===========")
                        print(f"切り取り画像 \n {clip_image}")
                        print(f"計算結果 \n {scalar}")
                        tmp_image[n, c, r, w] = scalar

        outc = tmp_image.sum(dim=1) # チャンネルごとに畳み込み結果を加算
        conv_image[:, o] = outc # 出力チャネル1個分の畳み込み結果を格納
        print(f"入力チャネル数ごとの畳み込み : {outc}")
    
    return conv_image

def main():
    # kernel2D()
    torch.manual_seed(42)

    batch_size = 2
    # dataloader_train = torch.utils.data.DataLoader(
    #     datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transforms.ToTensor()),
    #     batch_size=batch_size,
    #     shuffle=False
    # )
    
    n = 5
    c = 3
    s = 4
    image = torch.arange(n*c*s*s, dtype=torch.float).reshape(n, c, s, s)
    padding = 1
    stride = 2
    out_channels = 2
    
    fH = 3 
    fW = 3

    kernel_shape = (out_channels, c, fH, fW)
    print(f"画像表示")
    print(image)
    print(image.shape)
    print(kernel_shape)
    
    # kernel2D(kernel_shape, image) # 純粋な畳み込み
    zeroPadding(image, padding)
    
    # kernel2DP(kernel_shape, image, padding) # paddingを考慮した畳み込み
    
    conv_image = kernel2DPS(kernel_shape, image, padding, stride)
    print(f"畳み込み結果 : \n {conv_image}")
    
if __name__ == "__main__":
    main()