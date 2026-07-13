from utils import accuracy
import torch
from tqdm import tqdm
from custom_dataclasses import TrainerMetadata

class Trainer:

    def __init__(self, model, optimizer, criterion, device, trainer_config, recorder, classes):
        """ハイパーパラメータの初期化
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.trainer_config = trainer_config
        self.recorder = recorder
        self.classes = classes
        self.history = {
            "train" : {
                "accuracy" : [],
                "loss" : [],
                "l2_params": [],
                "l2_grads": [] 
            },
            
            "evaluate" : {
                "accuracy" : [],
                "loss" : [],
                "l2_params": [],
                "l2_grads": []
            }
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
    
    def fit(self, train_loader, validation_loader):
        """Train実行用関数
        """
        print(f"========Train and Evaluate Start!!!!========")
        for epoch in range(self.trainer_config["max_epochs"]):
            self.should_record = (epoch + 1) % self.trainer_config["verbose"] == 0
            self.current_epoch = epoch + 1
            
            # 学習
            score, loss, l2_params, l2_grads = self._run_epoch(
                train_loader,
                mode="train",
                desc=f"Train {self.current_epoch}/{self.trainer_config["max_epochs"]}"
            )
            
            self.history["train"]["accuracy"].append(score)
            self.history["train"]["loss"].append(loss)
            self.history["train"]["l2_params"].append(l2_params)
            self.history["train"]["l2_grads"].append(l2_grads)
            
            # 検証
            score, loss, l2_params, l2_grads = self._run_epoch(
                validation_loader,
                mode="evaluate",
                desc=f"Evaluate {self.current_epoch}/{self.trainer_config["max_epochs"]}"
            )
        
            self.history["evaluate"]["accuracy"].append(score)
            self.history["evaluate"]["loss"].append(loss)
            self.history["evaluate"]["l2_params"].append(l2_params)
            self.history["evaluate"]["l2_grads"].append(l2_grads)
            
        # for i, layer in enumerate(self.model.layers):
        #     print(f"===== layer ====== {layer.__class__.__name__}")
        #     for param, grad in zip(layer.params, layer.grads):
        #         # print(f"params : max {torch.max(torch.abs(param))}, min{torch.min(torch.abs(param))}, mean {torch.mean(torch.abs(param))} shape {param.shape}")
        #         # print(f"grads:  max {torch.max(torch.abs(grad))}, min{torch.min(torch.abs(grad))}, mean {torch.mean(torch.abs(grad))} shape {grad.shape}")
        #         print(f"更新率: {torch.mean(torch.abs(grad)) / torch.mean(torch.abs(param)) * 100}%")

    def prediction(self, test_loader):
        """Test実行用関数
        """
        print(f"========Test Start!!!!========")
        
        self.current_epoch = 0
        
        score, loss, last_x, last_t, last_pred = self._run_epoch(
            test_loader,
            mode="test",
            desc=f"Evaluate {self.current_epoch}/{self.trainer_config["max_epochs"]}"
        )
        
        return score, loss, last_x, last_t, last_pred
            
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
    
    def _step(self, x, t, mode="train"):
        if mode == "train":
            return self._train_step(x, t)
        return self._eval_step(x, t)

    def _run_epoch(self, dataloader, mode="train", desc=""):
        """1エポック分の処理内容を実装
        """
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_l2_params = 0
        epoch_l2_grads = 0
        pbar = tqdm(dataloader, desc=desc)
        cond = 1 == 0 
        
        for x, t in pbar:
            # CPU or CUDAをセット
            x = x.to(self.device)
            t = t.to(self.device)

            # 画像描画するラベルを取得
            # cond = self.trainer_config["plot_img_idx"] == t
            # plot_img_idx = cond.nonzero(as_tuple=True)[0][0].item() if len(cond.nonzero(as_tuple=True)[0]) > 0 else None
            # # 画像描画するラベルが含まれていれば、1epochに1度だけ記録する
            # # plot_img_idxにラベルが入っているとき、バッチ内に記録対象のバッチがあることを意味する
            # if plot_img_idx:
            #     with self.recorder.record(TrainerMetadata(self.model.name, mode, self.current_epoch, self.trainer_config["batch_size"], plot_img_idx), self.should_record):
            #         with self.model.hook_manager.register([self.recorder.forward_hook], self.should_record):
            #             # Train or Evaluate, Testの1エポック分の処理を実行
            #             pred, loss = self._step(x, t, mode)
            #             self.should_record = False
            # else:
            #     # Train or Evaluate, Testの1エポック分の処理を実行
            #     pred, loss = self._step(x, t, mode)
            
            pred, loss = self._step(x, t, mode)
            score = accuracy(pred, t)
            epoch_loss += loss
            epoch_accuracy += score

            pbar.set_postfix({
                "loss" : loss,
                "accuracy" : score
            })
        
        if mode in ["train", "evaluate"]:
            epoch_l2_params = sum([param.detach().clone().cpu().norm(p=2).item() for param in self.model.params])
            epoch_l2_grads = sum([grads.detach().clone().cpu().norm(p=2).item() for grads in self.model.grads])
            return epoch_accuracy / len(dataloader), epoch_loss / len(dataloader), epoch_l2_params, epoch_l2_grads
        
        if mode in ["test"]:
            last_x = x
            last_t = t
            last_pred = pred
            return epoch_accuracy / len(dataloader), epoch_loss / len(dataloader), last_x.detach().cpu(), last_t.detach().cpu(), last_pred