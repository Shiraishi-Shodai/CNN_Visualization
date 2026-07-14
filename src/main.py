import math
from layers import Affine, ReLU, SoftmaxWithLoss
import torch
from utils import softmax, cross_entropy_error, load_yaml, plot_imgsWithLabel, plot_epoch_metrics, view_confusion_matrix
from pathlib import Path
from sequential import Sequential
from optimizer import SGD
from torchvision import transforms, datasets
from trainer import Trainer
from model_builder import ModelBuilder
from torch.utils.data import Subset, random_split
from recorder import Recorder
from hook_manager import HookManager
from optimizer_builder import OptimizerBuilder
from visualizer import Visualizer

torch.manual_seed(42)

def main():
    
    generator = torch.Generator().manual_seed(42)

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
    
    test_data = datasets.CIFAR10(
        root="data/cifar10",
        train=False,
        download=True,
        transform=transform,
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 学習・検証データの準備
    train_data_size = len(train_data)
    train_subset = Subset(train_data, list(range( int(train_data_size * trainer_config["subset"])))) if "subset" in trainer_config.keys() else train_data
    train_subset_size = len(train_subset)
    train_size = int(train_subset_size * trainer_config["train_size"])
    valid_size = int(train_subset_size - train_size)
    
    train_dataset, valid_dataset = random_split(
        train_subset,
        [train_size, valid_size],
        generator=generator
    )

    print(f"trainサブセット : {train_subset_size}, 学習 : {train_size}, 検証 : {valid_size}")
    # print(f"trainサブセット : {train_subset_size}, 学習 : {len(train_dataset)}, 検証 : {len(valid_dataset)}")
    
    # テストデータの準備
    test_data_size = len(test_data)
    test_subset = Subset(test_data, list(range(int(test_data_size * trainer_config["subset"])))) if "subset" in trainer_config.keys() else test_data
    test_subset_size = len(test_subset)
    test_dataset = test_subset
    print(f"testサブセット : {test_subset_size}")
    
    # data loaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=trainer_config["batch_size"],
        shuffle=False
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=trainer_config["batch_size"],
        shuffle=False
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=trainer_config["batch_size"],
        shuffle=False
    )
    
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
    # Epochごとにplotする指標
    epoch_plots = trainer_config["epoch_plots"]
    
    # 学習・検証
    trainer.fit(train_loader=train_loader, validation_loader=valid_loader)
    
    # 学習と検証の混同行列の取得
    train_cm = trainer.history.get_evaluation_metrics("train").confusion_matrix
    valid_cm= trainer.history.get_evaluation_metrics("valid").confusion_matrix
    
    # 学習・検証結果の出力(Epochごと)
    plot_epoch_metrics(trainer.history.train, trainer.history.valid, epoch_plots, "public/img/train_valid_score_loss.png")
    # 学習・検証結果の出力(全Epochを通して)
    view_confusion_matrix(train_cm, valid_cm, classes, "public/img/train_valid_confusion_matrix.png")
    
    # テスト
    # score, loss, last_x, last_t, last_pred = trainer.prediction(test_loader)
    # view_images_num = 16
    
    # if len(last_t) < view_images_num:
    #     view_images_num = len(last_t)
    
    # correct_labels = [classes[label] for label in last_t]
    # pred_labels = [classes[label] for label in last_pred]
    # print(f"Accuracy : {score * 100 :.2f}%, Loss : {loss}")
    # plot_imgsWithLabel(data=last_x[:view_images_num], 
    #                    correct_labels=correct_labels[:view_images_num], 
    #                    pred_labels=pred_labels[:view_images_num], 
    #                    num_cols=4, 
    #                    save_filename="public/img/test_result.png",
    #                    axes_title=True,
    #                    title=""
    #                    )
    
if __name__ == "__main__":
    main()