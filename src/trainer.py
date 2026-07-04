from utils import accuracy
import torch
from tqdm import tqdm
from custom_dataclasses import TrainerMetadata

class Trainer:

    def __init__(self, model, optimizer, criterion, device, trainer_config, recorder):
        """ハイパーパラメータの初期化
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.trainer_config = trainer_config
        self.recorder = recorder
        self.history = {
            "accuracy" : [],
            "loss" : []
        }
        self.should_record = False
        self.current_epoch = None
    
    # def fit(self, train_loader, max_epochs=100, verbose=100):
    #     """学習処理
    #     Parameters
    #     ----------
    #     train_loader : torch.util.data.DataLoader的なもの
    #     max_epochs : 最大epoch数
    #     verbose : printする間隔
    #     """
        
    #     for epoch in range(max_epochs):
    #         epoch_accuracy  = 0
    #         epoch_loss = 0
    #         for x, t in train_loader:
    #             x = x.to(self.device)
    #             t = t.to(self.device)

    #             pred, loss = self._train_step(x, t)
    #             epoch_accuracy  += accuracy(pred, t)
    #             epoch_loss += loss
    #             print(loss, accuracy(pred, t))
            
    #         avg_accuracy = epoch_accuracy  / len(train_loader)
    #         avg_loss = epoch_loss / len(train_loader)
    #         self.history["accuracy"].append(avg_accuracy)
    #         self.history["loss"].append(avg_loss)
    
    #         if (epoch + 1) % verbose == 0:
    #             print(f"Epoch [{epoch + 1}/{max_epochs}] Accuracy : {avg_accuracy}% Loss : {avg_loss}")
    
    def fit(self, train_loader):
        """Train実行用関数
        """
        for epoch in range(self.trainer_config["max_epochs"]):
            self.should_record = (epoch + 1) % self.trainer_config["verbose"] == 0
            self.current_epoch = epoch + 1
            score, loss = self._run_epoch(
                train_loader,
                train=True,
                desc=f"Train {self.current_epoch}/{self.trainer_config["max_epochs"]}"
            )
            
            self.history["accuracy"].append(score)
            self.history["loss"].append(loss)
    
    def evaluate(self, validation_loader, mode="Test"):
        """Evaluate, Test実行用関数
        """
        self.should_record = True
        score, loss = self._run_epoch(
            validation_loader,
            train=False,
            desc=f"{mode} mode"
        )
        
        return score, loss
            
    def _train_step(self, x, t):
        """"1回分のTrain処理を実装(パラメータ更新あり)
        """
        # モデルの順伝搬
        y = self.model.forward(x)
        # 予測値取得(Affine出力の2次元行列 → ラベル行列, Softmaxは通っていないが、値が大きさはSoftmaxを通っても変わらないためAffineの出力から最大値を取っても問題なし)
        pred = y.argmax(dim=1)
        # 損失の計算
        loss = self.criterion.forward(y, t)
        # 損失からの勾配を取得 (cross entropy → softmax → dout)
        dout = self.criterion.backward()
        # モデルの逆伝搬(勾配の計算)
        self.model.backward(dout)
        # パラメータの更新
        self.optimizer.update(self.model.params, self.model.grads)
        
        return pred, loss
    
    def _eval_step(self, x, t):
        """1回分のEvaluate, Test処理を実行(パラメータ更新なし)
        """
        # モデルの順伝搬
        y = self.model.forward(x)
        # 予測値取得(Affine出力の2次元行列 → ラベル行列, Softmaxは通っていないが、値が大きさはSoftmaxを通っても変わらないためAffineの出力から最大値を取っても問題なし)
        pred = y.argmax(dim=1)
        # 損失の計算
        loss = self.criterion.forward(y, t)
        
        return pred, loss

    def _run_epoch(self, dataloader, train=True, desc=""):
        """1エポック分の処理内容を実装
        """
        epoch_loss = 0
        epoch_accuracy = 0
        metadata = None
        mode = "train" if train else "eval"
        pbar = tqdm(dataloader, desc=desc)
        
        for x, t in pbar:
            # CPU or CUDAをセット
            x = x.to(self.device)
            t = t.to(self.device)
            
            with self.recorder.record(TrainerMetadata("VGG16", mode, self.current_epoch, self.trainer_config["batch_size"]), self.should_record):
                if self.should_record:
                    self.model.hook_manager.register_all_forward_hooks([self.recorder.forward_hook])
                # Train or Evaluate, Testの1エポック分の処理を実行
                if train:
                    pred, loss = self._train_step(x, t)
                else:
                    pred, loss = self._eval_step(x, t)

                self.should_record = False
                
                if self.should_record:
                    self.model.hook_manager.unregister_all_forward_hooks()
                
            
            score = accuracy(pred, t)
            epoch_loss += loss
            epoch_accuracy += score

            pbar.set_postfix({
                "loss" : loss,
                "accuracy" : score
            })
        
        return epoch_accuracy / len(dataloader), epoch_loss / len(dataloader)