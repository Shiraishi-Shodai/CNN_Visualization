from hook_manager import HookManager
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
    def __init__(self, layers : list):
        self.layers = layers
        self.device = None
        self.hooks = HookManager()
    
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
        for layer in self.layers:
            x = layer.forward(x)
            
            self.hooks.call_forward_hooks(layer, x)
        return x
    
    def backward(self, dout=1):
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout