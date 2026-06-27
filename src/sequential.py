
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
        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
    def backward(self, dout=1):
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout