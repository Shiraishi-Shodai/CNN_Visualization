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
    Parameters
    ----------
    layer_cfg : layerのconfig
    fan_in : 入力値の次元数(ニューロン数)
    
    Returns
    -------
    layer : 生成したlayer
    """
    fan_out = fan_in
    return ReLU(), fan_out

@register("Convolution")
def build_convolution(layer_cfg, fan_in):
    """Convolutionを生成する
    Parameters
    ----------
    layer_cfg : layerのconfig
    fan_in : 入力値の次元数(ニューロン数)
    
    Returns
    -------
    layer : 生成したlayer
    """
    kernel = (layer_cfg["out_channels"], layer_cfg["in_channels"], layer_cfg["kernel_h"], layer_cfg["kernel_w"])
    stride = layer_cfg["stride"]
    padding = layer_cfg["padding"]

    fan_out = layer_cfg["out_features"]
    weight_init = get_initializer(layer_cfg["initializer"]["weight"])
    bias_init = get_initializer(layer_cfg["initializer"]["bias"])

    weight = weight_init(kernel, fan_in, fan_out)
    bias = bias_init((layer_cfg["out_channels"], ))
    
    return Convolution(weight, bias, stride, padding), fan_out

@register("MaxPooling")
def build_max_pooling(layer_cfg, fan_in):
    """MaxPoolingを生成する
    Parameters
    ----------
    layer_cfg : layerのconfig
    fan_in : 入力値の次元数(ニューロン数)
    
    Returns
    -------
    layer : 生成したlayer
    """

    fan_out = layer_cfg["out_features"]
    pool_h = layer_cfg["pool_h"]
    pool_w = layer_cfg["pool_w"]
    stride = layer_cfg["stride"]
    padding = layer_cfg["padding"]
    
    return MaxPooling(pool_h, pool_w, stride, padding), fan_out

@register("Flatten")
def build_relu(layer_cfg, fan_in):
    """Flattenを生成する
    Parameters
    ----------
    layer_cfg : layerのconfig
    fan_in : 入力値の次元数(ニューロン数)
    
    Returns
    -------
    layer : 生成したlayer
    """
    fan_out = layer_cfg["out_features"]
    return Flatten(), fan_out