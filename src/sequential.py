from hook_manager import HookManager
from custom_dataclasses import LayerRecord
class Sequential:
    """生成したレイヤーをリストで受け取り、モデルをインスタンス化するためのクラス
        YAML
            │
            ▼
    ModelBuilder
            │
    List[Layer]
            │
            ▼
    Sequential
    ├── forward()
    ├── backward()
    ├── params
    └── grads

    Trainer
    ├── model.forward(x)
    ├── loss.forward(y, t)
    ├── loss.backward()
    ├── model.backward(dout)
    └── optimizer.update(...)
    """
    def __init__(self, name : str, layers : list, hook_manager : HookManager):
        self.name = name
        self.layers = layers
        self.device = None
        self.hook_manager = hook_manager

    @property
    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params
        
        return params

    @property
    def grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.grads
        
        return grads
    
    def to(self, device):
        """各レイヤーにdeviceをセットする
        """
        self.device = device
        
        for layer in self.layers:
            for i in range(len(layer.params)):
                layer.params[i] = layer.params[i].to(device)
                layer.grads[i] = layer.grads[i].to(device)
    
    def forward(self, x):
        ctx = None
        y = None
        for layer in self.layers:
            y = layer.forward(x)
            # print(f"sequential : {x.shape, y.shape}")
            self.hook_manager.call_forward_hooks(layer.__class__.__name__, x.detach().clone().cpu(), y.detach().clone().cpu())
            x = y
        return x
    
    def backward(self, dout=1):
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout