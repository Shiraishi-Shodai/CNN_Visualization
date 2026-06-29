
class HookManager:
    def __init__(self):
        self.forward_hooks = []
    
    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)
    
    def call_forward_hooks(self, layer, output):
        """順伝搬時のhookをすべて呼び出す
        Parameter
        ---------
        layer : モデルのレイヤー
        output : レイヤーの出力
        
        """
        for hook in self.forward_hooks:
            hook(layer, output)