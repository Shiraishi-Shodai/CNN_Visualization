import math
from layers import Affine, ReLU, SoftmaxWithLoss
import torch
from utils import softmax, cross_entropy_error, load_yaml
from pathlib import Path
from sequential import Sequential
from optimizer import SGD
from torchvision import transforms, datasets
from trainer import Trainer
from model_builder import ModelBuilder
from torch.utils.data import Subset

torch.manual_seed(42)

def main():
    
    # deviceの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス {device}")    

    # ハイパーパラメータの読み込み
    model_config = load_yaml("config/VGG16.yaml")
    train_config = load_yaml("config/train.yaml")["train"]

    # transformの定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    # データの読み込み
    train_data = datasets.CIFAR10(
        root="data/cifar10", 
        train=True, 
        download=True, 
        transform=transform,
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    train_subset = Subset(train_data, list(range(train_config["subset"]))) if train_config["sample"] else train_data
    # data loaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=train_config["batch_size"],
        shuffle=True
        )
    
    print(f"使用する学習データ数 {len(train_loader.dataset)}") # データ数

    # sample = next(iter(train_loader))
    # sample_x, sample_t = sample[0], sample[1]
    
    # Trainerの定義
    mb = ModelBuilder(model_config)
    layers = mb.build()
    model = Sequential(layers)
    model.to(device)
    optimizer = SGD(train_config["lr"])
    criterion = SoftmaxWithLoss()
    trainer = Trainer(model, optimizer, criterion, device)
    
    # 学習
    trainer.fit(train_loader=train_loader, max_epochs=train_config["max_epochs"])


if __name__ == "__main__":
    main()