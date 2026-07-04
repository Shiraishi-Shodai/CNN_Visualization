# trainer:
#   recorder:
#     enable: true
#     save_forward: true
#     save_backward: true
#     max_samples: 10
from utils import plot_imgs
from custom_dataclasses import LayerRecord
import torch
from contextlib import contextmanager
class Recorder:
    """
    順伝搬, 逆伝搬用のデータを記録する
    """
    def __init__(self):
        self.forward_feature_maps = []
        self.PLOT_NUM = 5 # 横方向にプロットする画像サイズ
        self.trainer_metatdata = None
    
    def forward_hook(self, layer_name , input_tensor, output_tensor):
        """input画像は3チャンネル。その他の画像はチャンネル方向に平均した画像を格納
        """
        ctx = LayerRecord(
            layer_name=layer_name,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            input_shape=input_tensor.shape,
            output_shape=output_tensor.shape
        )
        self.forward_feature_maps.append(ctx)
        # print(f"順伝搬{layer.__class__.__name__, ctx.inputs.shape, ctx.outputs.shape}")

    def backward_hook(self, layer, dout):
        print(f"逆伝搬{layer, dout.shape}")
        
    def begin_forward(self, metadata):
        self.forward_feature_maps = []
        self.trainer_metadata = metadata
    
    def end_forward(self):
        torch.save(self.forward_feature_maps, rf"public/pt/{self.trainer_metadata.model_name}_{self.trainer_metadata.mode}_epoch{self.trainer_metadata.epoch}_batch{self.trainer_metadata.batch}.pt")
        self.forward_feature_maps = []
    
    @contextmanager
    def record(self, metadata, enabled=True):
        if not enabled:
            yield
            return
        self.begin_forward(metadata)
        try:
            yield
        finally:
            self.end_forward()