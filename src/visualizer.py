from utils import plot_imgs
import torch

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
            
        plot_imgs(layer_records, num_cols, save_filename)

    def featureMapViewer(self):
        pass
    
    
v = Visualizer()
layer_records = torch.load(r"C:/Users/siran/ML/cnn_visualization/public/pt/VGG16_train_epoch2_batch1.pt", weights_only=False)

v.overviewViewer(layer_records, 5, r"public/img/vgg16_sample.png")