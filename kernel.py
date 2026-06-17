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
    padding=0
    
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
        kernel_shape : (C=1, H, W)
    """
    C, H, W = image.shape
    
    # paddingが0以下なら処理を終了する
    if padding <= 0 :
        return image
    
    padding_image = torch.zeros(C, H + padding * 2, W + padding * 2, dtype=torch.float)
    
    for c in range(C):
        padding_image[c, padding : -padding , padding : -padding] = image[c]
    
    return padding_image

def main():
    # kernel2D()
    torch.manual_seed(42)

    batch_size = 2
    # dataloader_train = torch.utils.data.DataLoader(
    #     datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transforms.ToTensor()),
    #     batch_size=batch_size,
    #     shuffle=False
    # )
    
    image = torch.arange(4*4, dtype=torch.float).reshape(1, 4, 4)
    padding = 1
    print(f"画像表示")
    print(image)
    kernel_shape = (1, 3, 3)
    
    # kernel2D(kernel_shape, image) # 純粋な畳み込み
    # zeroPadding(image, padding)
    
    # kernel2DP(kernel_shape, image, padding) # paddingを考慮した畳み込み
    
if __name__ == "__main__":
    main()