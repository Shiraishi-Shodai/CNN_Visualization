import math
from layers import Affine, ReLU, SoftmaxWithLoss
import torch
from utils import *
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
from experiment_manager import ExperimentManager
from checkpoint_manager import CheckPointManager

torch.manual_seed(42)

def main():
    
    generator = torch.Generator().manual_seed(42)

    # deviceの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス {device}")    

    # ハイパーパラメータの読み込み
    # model_config = load_yaml("config/miniVGG.yaml")
    model_config = load_yaml("config/miniCNN.yaml")
    trainer_config = load_yaml("config/train.yaml")["train"]
    optimizer_config = load_yaml("config/optimizer.yaml")

    # transformの定義
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=(0.4914, 0.4822, 0.4465),
        # std=(0.2470, 0.2435, 0.2616)
        # ),
        # transforms.Resize((32, 32)),
    ])

    test_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])
    
    # データの読み込み
    full_train_data = datasets.CIFAR10(
        root="data/cifar10", 
        train=True, 
        download=True, 
        transform=train_transform,
    )
    
    test_data = datasets.CIFAR10(
        root="data/cifar10",
        train=False,
        download=True,
        transform=test_transform,
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 学習・検証データの準備
    train_data_size = len(full_train_data)
    train_valid_subset = Subset(full_train_data, list(range( int(train_data_size * trainer_config["subset"])))) if "subset" in trainer_config.keys() else full_train_data
    train_valid_subset_size = len(train_valid_subset)
    train_size = int(train_valid_subset_size * trainer_config["train_size"])
    valid_size = int(train_valid_subset_size - train_size)
    
    train_indices, valid_indices = random_split(
        range(train_valid_subset_size),
        [train_size, valid_size],
        generator=generator
    )
    
    train_data = datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        transform=train_transform
    )

    valid_data = datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        transform=test_transform
    )

    train_dataset = Subset(train_data, train_indices.indices)
    valid_dataset = Subset(valid_data, valid_indices.indices)
    
    print(f"trainサブセット : {train_valid_subset_size}, 学習 : {train_size}, 検証 : {valid_size}")
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
    # モデルのデバイスをセット
    model.to(device)
    # optimizerのビルド
    ob = OptimizerBuilder(optimizer_config)
    optimizer = ob.build()
    # Epochごとにplotする指標
    epoch_plots = trainer_config["epoch_plots"]
    # 学習記録管理用
    ex_manager = ExperimentManager()
    # 学習記録管理内のモデル保存・取り出し用
    ch_manager = CheckPointManager()
    # trainerのビルド
    trainer = Trainer(model, optimizer, criterion, device, trainer_config, recorder, classes, ex_manager, ch_manager)
    
    # 学習・検証
    trainer.fit(train_loader=train_loader, validation_loader=valid_loader)
    
    # 実験結果の保存
    trainer.save()
    
    # 各データのラベル分布を表示
    train_labels = torch.bincount(torch.tensor(get_dataset_labels(train_dataset)), minlength=trainer_config["class_num"])
    valid_labels = torch.bincount(torch.tensor(get_dataset_labels(valid_dataset)), minlength=trainer_config["class_num"])
    test_labels = torch.bincount(torch.tensor(get_dataset_labels(test_dataset)), minlength=trainer_config["class_num"])
    view_label_distribution(train_labels, valid_labels, test_labels, classes, fr"{trainer.ex_manager.experiment_sub_dir.imgs}/label_distribute.png")
    
    # 学習と検証の混同行列の取得
    train_cm = trainer.history.get_evaluation_metrics("train").confusion_matrix
    valid_cm= trainer.history.get_evaluation_metrics("valid").confusion_matrix
    
    # 学習・検証のクラスごとの正解率を取得
    train_class_accuracy = confusion_matrix_calc(train_cm, "class accuracy")
    valid_class_accuracy = confusion_matrix_calc(valid_cm, "class accuracy")
    
    # 学習・検証結果の出力(Epochごと)
    plot_epoch_metrics(trainer.history.train, trainer.history.valid, epoch_plots, fr"{trainer.ex_manager.experiment_sub_dir.imgs}/train_valid_score_loss.png")
    
    # 学習・検証結果の出力(全Epochを通して)
    view_confusion_matrix(train_cm, valid_cm, classes, fr"{trainer.ex_manager.experiment_sub_dir.imgs}/train_valid_confusion_matrix.png")
    view_class_accuracy(train_class_accuracy, valid_class_accuracy, classes, fr"{trainer.ex_manager.experiment_sub_dir.imgs}/train_valid_class_accuracy.png")

    # テスト
    last_x, last_t, last_pred = trainer.prediction(test_loader)
    view_images_num = 16
    test_score = trainer.history.test[0].accuracy
    test_loss = trainer.history.test[0].loss
    print(f"Accuracy : {test_score * 100 :.2f}%, Loss : {test_loss}")
    
    # 最後のバッチの予測結果を確認
    if len(last_t) < view_images_num:
        view_images_num = len(last_t)
    
    correct_labels = [classes[label] for label in last_t]
    pred_labels = [classes[label] for label in last_pred]
    plot_imgsWithLabel(data=last_x[:view_images_num], 
                       correct_labels=correct_labels[:view_images_num], 
                       pred_labels=pred_labels[:view_images_num], 
                       num_cols=4, 
                       save_filename=fr"{trainer.ex_manager.experiment_sub_dir.imgs}/test_result.png",
                       axes_title=True,
                       title=""
                       )
    
if __name__ == "__main__":
    main()