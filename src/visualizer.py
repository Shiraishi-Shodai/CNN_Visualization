from utils import plot_imgs
import torch
from copy import deepcopy

class Visualizer:
    def overviewViewer(self, layer_records: list, num_cols : int, save_filename : str) -> None:
        """"各レイヤーの出力画像を描画。1枚の画像を保存。4チャンネル以上の画像はチャンネル方向に平均化した後プロット
        Parameter
        ---------
        layer_records : list[layer_record] layer_record.input or output (N, C, H, W or N, M)
        num_cols : 横一列に表示する画像枚数
        save_filename : ファイル名。ディレクトリパスあり(r"public/img/vgg16_sample_overview.png")
        """
        
        # 入出力が(N, C, H, W)のレイヤーのみにフィルター
        layer_records = [layer_record for layer_record in layer_records if len(layer_record.input_shape) > 2 and len(layer_record.output_shape) > 2]
            
        plot_imgs(layer_records, num_cols, save_filename, axes_title=True, title=None)

    def featureMapViewer(self, layer_records: list, num_cols : int, save_filename : str) -> None:
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
        dir_path = "/".join(save_filename.split("/")[:-1])
        file_name = save_filename.split("/")[-1]
        
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
            save_filename = rf"{dir_path}/layer_{i}_{file_name}"
            plot_imgs(dummy_layer_records, num_cols, save_filename, axes_title=False, title=f"layer_{i}_{dummy_layer_record.name}")
    
v = Visualizer()
layer_records = torch.load(r"C:/Users/siran/ML/cnn_visualization/public/pt/VGG16_train_epoch2_batch1.pt", weights_only=False)

v.overviewViewer(layer_records, 5, r"public/img/vgg16_sample_overview.png")
# v.featureMapViewer(layer_records, 5, r"public/img/featuremap_vgg16/sample.png")