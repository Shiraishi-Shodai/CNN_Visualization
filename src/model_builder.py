from utils import load_yaml
from pathlib import Path
from initializers import get_initializer
from layers import *

class ModelBuilder:
    """configで受け取ったレイヤーを生成し、生成したレイヤーのリストを返す。
       forwardやbackward, params, gradの管理はSequentialで行う
    """
    
    def __init__(self, config):
        self.config = config
    
    def build(self):
        """
        │
        ├── input_size を取得
        │
        ├── layers = []
        │
        ├── YAMLのlayersを順番に読む
        │      │
        │      ├── Affineなら
        │      │      ├── out_features取得
        │      │      ├── Initializer取得
        │      │      ├── Weight生成
        │      │      ├── Bias生成
        │      │      ├── Affine生成
        │      │      ├── layersへ追加
        │      │      └── fan_in更新
        │      │
        │      ├── ReLUなら
        │      │      └── layersへ追加
        │      │
        │      └── ...
        │
        └── layersを返す
        """
        
        model_cfg = self.config["model"]
        layers = []
        fan_in = model_cfg["input_size"]

        for layer_cfg in model_cfg["layers"]:
            layer_type = layer_cfg["type"]
            
            if layer_type == "Affine":
                layer, fan_out = self._build_affine(layer_cfg, fan_in)
                
                layers.append(layer)
                fan_in = fan_out
            
            elif layer_type == "ReLU":
                layer = self._build_relu()  
                layers.append(layer)
            else:
                raise ValueError(
                    f"layer type : {layer_type}が見つかりません"
                )
        
        return layers
    
    def _build_affine(self, layer_cfg, fan_in):
        """Affineを生成する
        Parameters
        ----------
        layer_cfg : layerのconfig
        fan_in : 入力値の次元数(ニューロン数)
        
        Returns
        -------
        layer : 生成したlayer
        fan_out : 次のlayerのfan_in
        """
        fan_out = layer_cfg["out_features"]
        weight_name = layer_cfg["initializer"]["weight"]
        bias_name = layer_cfg["initializer"]["bias"]

        weight_init = get_initializer(weight_name)
        bias_init = get_initializer(bias_name)

        weight = weight_init(
            (fan_in, fan_out),
            fan_in,
            fan_out
        )
        
        bias = bias_init(
            (fan_out, )
        )
        
        layer =  Affine(weight, bias)
        return layer, fan_out
    
    def _build_relu(self):
        """ReLUを生成する
        Returns
        -------
        layer : 生成したlayer
        """
        return ReLU()

yaml_file = Path(r"config/simple_dense2.yaml")
config = load_yaml(yaml_file)

mb = ModelBuilder(config)

layers = mb.build()
print(layers)