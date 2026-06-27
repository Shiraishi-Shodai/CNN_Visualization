from initializer_registry import *
from layers import *

LAYER_BUILDERS = {}

def register(name):
    def decorator(func):
        LAYER_BUILDERS[name] = func
        return func

    return decorator

def get_layer_builder(name):
    if name not in LAYER_BUILDERS:
        raise ValueError(f"{name}はlayer registerに登録されていません")

    return LAYER_BUILDERS[name]

@register("Affine")
def build_affine(layer_cfg, fan_in):
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

@register("ReLU")
def build_relu(layer_cfg, fan_in):
    """ReLUを生成する
    Returns
    -------
    layer : 生成したlayer
    """
    fan_out = fan_in
    return ReLU(), fan_out