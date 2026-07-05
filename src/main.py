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
from recorder import Recorder
from hook_manager import HookManager
from optimizer_builder import OptimizerBuilder
from visualizer import Visualizer

torch.manual_seed(42)

def main():
    
    # deviceの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス {device}")    

    # ハイパーパラメータの読み込み
    model_config = load_yaml("config/miniVGG.yaml")
    trainer_config = load_yaml("config/train.yaml")["train"]
    optimizer_config = load_yaml("config/optimizer.yaml")

    # transformの定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])
    
    # データの読み込み
    train_data = datasets.CIFAR10(
        root="data/cifar10", 
        train=True, 
        download=True, 
        transform=transform,
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    train_subset = Subset(train_data, list(range(trainer_config["subset"]))) if trainer_config["sample"] else train_data
    # data loaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=trainer_config["batch_size"],
        shuffle=False
        )
    
    print(f"使用する学習データ数 {len(train_loader.dataset)}") # データ数

    # 損失関数の定義
    criterion = SoftmaxWithLoss()
    recorder = Recorder()
    hook_manager = HookManager()
    # モデルのビルド
    mb = ModelBuilder(model_config)
    layers = mb.build()
    model = Sequential(model_config["model"]["name"], layers, hook_manager)
    model.to(device)
    # optimizerのビルド
    ob = OptimizerBuilder(optimizer_config)
    optimizer = ob.build(model.params)
    # trainerのビルド
    trainer = Trainer(model, optimizer, criterion, device, trainer_config, recorder, classes)
    # 学習
    trainer.fit(train_loader=train_loader)
    
    visualizer = Visualizer(history=trainer.history, view_mode="plot_fit_history", save_filename=r"public/img/history.png")
    visualizer.view()
    
if __name__ == "__main__":
    main()