from utils import plot_imgs
import torch
from copy import deepcopy

class Visualizer:
    def overviewViewer(self, layer_records: list, num_cols : int, save_filename : str) -> None:
        layer_records = [layer_record for layer_record in layer_records if len(layer_record.input_shape) > 2 and len(layer_record.output_shape) > 2]
        for i, layer_record in enumerate(layer_records):
            # print(layer_records[i].name, layer_records[i].input_shape, layer_records[i].output_shape, len(layer_records[i].input_shape), len(layer_records[i].output_shape))
            
            input_channels = layer_record.input_shape[0]
            output_channels = layer_record.output_shape[0]
            if input_channels > 3:
                layer_record.input_tensor = layer_record.input_tensor.mean(dim=0, keepdims=True)
            if input_channels > 3:
                layer_record.output_tensor = layer_record.output_tensor.mean(dim=0, keepdims=True)
            
        plot_imgs(layer_records, num_cols, save_filename, axes_title=True, title=None)

    def featureMapViewer(self, layer_records: list, num_cols : int, save_filename : str) -> None:
        layer_records = [layer_record for layer_record in layer_records if len(layer_record.input_shape) > 2 and len(layer_record.output_shape) > 2]
        dir_path = "/".join(save_filename.split("/")[:-1])
        file_name = save_filename.split("/")[-1]
        
        for i, layer_record in enumerate(layer_records):
            dummy_layer_records = []
            dummy_layer_record = deepcopy(layer_record)
            output_channels = layer_record.output_shape[1]
            for output_channel in range(output_channels):
                # print(layer_record.output_shape, output_channel, layer_record.output_tensor[:, output_channel].shape)
                dummy_layer_record.output_tensor = deepcopy(layer_record.output_tensor[:, output_channel:output_channel+1]) # output_channel:output_channel+1でkeepdims的な効果を期待
                dummy_layer_records.append(dummy_layer_record)
            save_filename = rf"{dir_path}/layer_{i}_{file_name}"
            plot_imgs(dummy_layer_records, num_cols, save_filename, axes_title=False, title=f"layer_{i}_{dummy_layer_record.name}")
    
v = Visualizer()
layer_records = torch.load(r"C:/Users/siran/ML/cnn_visualization/public/pt/VGG16_train_epoch2_batch1.pt", weights_only=False)

# v.overviewViewer(layer_records, 5, r"public/img/vgg16_sample_overview.png")
v.featureMapViewer(layer_records, 5, r"public/img/featuremap_vgg16/sample.png")