# trainer:
#   recorder:
#     enable: true
#     save_forward: true
#     save_backward: true
#     max_samples: 10
from utils import plot_imgs
class Recorder:
    """
    順伝搬, 逆伝搬用のデータを記録する
    """
    def __init__(self):
        self.forward_feature_maps = {}
        self.PLOT_NUM = 5 # 横方向にプロットする画像サイズ
    
    def forward_hook(self, layer, ctx):
        """input画像は3チャンネル。その他の画像はチャンネル方向に平均した画像を格納
        """
        class_name = layer.__class__.__name__
        
        if len(self.forward_feature_maps) == 0:
            self.forward_feature_maps["Input"] = ctx.inputs
        
        self.forward_feature_maps[class_name] = ctx.outputs.mean(dim=1)
        # print(f"順伝搬{layer.__class__.__name__, ctx.inputs.shape, ctx.outputs.shape}")

    def backward_hook(self, layer, dout):
        print(f"逆伝搬{layer, dout.shape}")
        
    
    def plot_forward_feature_map(self):
        pass

    def forward_plot(self):
        # plot_imgs(self.forward_feature_maps)
        pass