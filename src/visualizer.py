from utils import plot_imgs
import torch
from copy import deepcopy
import glob
from pathlib import Path
from matplotlib import pyplot as plt
import math
class Visualizer:
    def __init__(self, history=None, pt_dir_path=None, save_dir_path=None, view_mode="", num_cols=5, save_filename=""):
        self.pt_dir_path = pt_dir_path
        self.save_dir_path = save_dir_path
        self.view_mode = view_mode
        self.num_cols = num_cols
        self.history = history
        self.save_filename = save_filename
        self.view_function = None
    
    def view(self):
        match self.view_mode:
            case "overview":
                self.view_function = self._overviewViewer
            case "featureview":
                self.view_function = self._featureMapViewer
            case "plot_fit_history":
                self.view_function = self._plot_fit_history
            case _:
                raise ValueError(f"{self.view_mode}は存在しません")
        
        if self.view_mode in ["overview", "featureview"]:
            file_list = list(Path(self.pt_dir_path).glob("*.pt"))
            for file in file_list:
                layer_records = torch.load(file, weights_only=False)
                save_filename = Path(self.save_dir_path, file.name.replace(".pt", ".png"))
                self.view_function(layer_records, save_filename)
        
        if self.view_mode == "plot_fit_history":
            self.view_function()
        
    def _overviewViewer(self, layer_records: list, save_filename : str) -> None:
        """"各レイヤーの出力画像を描画。1枚の画像を保存。4チャンネル以上の画像はチャンネル方向に平均化した後プロット
        Parameter
        ---------
        layer_records : list[layer_record] layer_record.input or output (N, C, H, W or N, M)
        num_cols : 横一列に表示する画像枚数
        save_filename : ファイル名。ディレクトリパスあり(r"public/img/vgg16_sample_overview.png")
        """
        
        # 入出力が(N, C, H, W)のレイヤーのみにフィルター
        layer_records = [layer_record for layer_record in layer_records if len(layer_record.input_shape) > 2 and len(layer_record.output_shape) > 2]
            
        plot_imgs(layer_records, self.num_cols, save_filename, axes_title=True, title=None)

    def _featureMapViewer(self, layer_records: list, save_filename : Path) -> None:
        """"各レイヤーの出力画像を描画。各レイヤーごとに画像を保存。画像内には、レイヤーの各チャンネル画像をグレースケールで描画
        Parameter
        ---------
        layer_records : list[layer_record] layer_record.input or output (N, C, H, W or N, M)
        num_cols : 横一列に表示する画像枚数
        save_filename : ファイル名。ディレクトリパスあり(r"public/img/vgg16_sample_overview.png")
        """
        
        # 入出力が(N, C, H, W)のレイヤーのみにフィルター
        layer_records = [layer_record for layer_record in layer_records if len(layer_record.input_shape) > 2 and len(layer_record.output_shape) > 2]
        
        # ディレクトリパスとファイル名を取得
        dir_path = save_filename.parent
        file_name = save_filename.name
        middle_path = None
        # レイヤー数分ループ
        for i, layer_record in enumerate(layer_records):
            dummy_layer_records = []
            dummy_layer_record = deepcopy(layer_record)
            output_channels = layer_record.output_shape[1]
            
            # チャンネル数分ループ
            for output_channel in range(output_channels):
                # print(layer_record.output_shape, output_channel, layer_record.output_tensor[:, output_channel].shape)
                dummy_layer_record.output_tensor = deepcopy(layer_record.output_tensor[:, output_channel:output_channel+1]) # output_channel:output_channel+1でkeepdims的な効果を期待
                dummy_layer_records.append(dummy_layer_record)
            
            middle_path = Path(f"layer_{i}_")
            save_filename = Path(self.save_dir_path, middle_path, file_name)
            plot_imgs(dummy_layer_records, self.num_cols, save_filename, axes_title=False, title=f"layer_{i}_{dummy_layer_record.name}")
    
    def _plot_fit_history(self):
        if len(self.history.keys()) == 0:
            raise ValueError(f"history key is not found")
        
        rows = math.ceil(len(self.history.keys()) / 2)
        fig, axes = plt.subplots(rows, 2)
        key_list = list(self.history.keys())
        
        for r in range(rows):
            for c in range(2):
                
                key = key_list[r * 2 + c]
                value = self.history[key]
                axes[r, c].plot(range(len(value)), value)
                axes[r, c].set_title(key)
        
        fig.tight_layout() # 各axisが重ならないように設定
        plt.savefig(self.save_filename)
# pt_dir_path = Path(r"public/pt/miniVGG")
# save_dir_path = Path(r"public/img/miniVGG")
# v = Visualizer(pt_dir_path, save_dir_path, "overview", 5)
# v.view()
# v.featureMapViewer(layer_records, 5, r"public/img/featuremap_vgg16/sample.png")