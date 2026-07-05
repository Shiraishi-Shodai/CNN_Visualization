from utils import plot_imgs
from custom_dataclasses import LayerRecord, GradientRecord, Record
import torch
from contextlib import contextmanager
from matplotlib import pyplot as plt
class Recorder:
    """
    順伝搬, 逆伝搬用のデータを記録する
    """
    def __init__(self):
        self.forward_feature_maps = []
        self.gradient_record_list = []
        self.trainer_metatdata = None
        # self.record = Record()
    
    def forward_hook(self, layer_name , input_tensor, output_tensor):
        """input画像は3チャンネル。その他の画像はチャンネル方向に平均した画像を格納
        """
        # print(f"recorder shape : {input_tensor.shape, output_tensor.shape}")
        input_tensor = input_tensor[self.trainer_metadata.plot_img_idx:self.trainer_metadata.plot_img_idx+1]
        output_tensor = output_tensor[self.trainer_metadata.plot_img_idx:self.trainer_metadata.plot_img_idx+1]
        
        ctx = LayerRecord(
            name=layer_name,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            input_shape=input_tensor.shape,
            output_shape=output_tensor.shape
        )

        self.forward_feature_maps.append(ctx)

    def backward_hook(self, layer_name , grads):
        
        ctx = GradientRecord(
            name=layer_name,
            grad_norm=grads.norm(p=2).item(),
            grad_std=grads.std().item(),
            grad_mean=grads.mean().item(),
            grad_max=grads.max().item(),
            grad_min=grads.min().item()
        )
        
    def begin_forward(self, metadata):
        """forward_hook前に実行する処理
        """
        self.forward_feature_maps = []
        self.trainer_metadata = metadata
    
    def end_forward(self):
        """forward_hook後に実行する処理
        """
        torch.save(self.forward_feature_maps, rf"public/pt/{self.trainer_metadata.name}/{self.trainer_metadata.mode}_epoch{self.trainer_metadata.epoch}_batch{self.trainer_metadata.batch}.pt")
        self.forward_feature_maps = []
    
    @contextmanager
    def record(self, metadata, enabled=True):
        """contextmanagerで準備と後かたずけセットで登録。enabledがFalseなら準備も後かたずけもしない
        """
        if not enabled:
            yield
            return
        self.begin_forward(metadata)
        try:
            yield
        finally:
            self.end_forward()