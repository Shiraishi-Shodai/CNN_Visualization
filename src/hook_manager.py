from contextlib import contextmanager
class HookManager:
    def __init__(self):
        self.forward_hooks = []
    
    def _register_forward_hook(self, hook):
        self.forward_hooks.append(hook)

    def _register_backward_hook(self, hook):
        self.backward_hooks.append(hook)
    
    def register_all_forward_hooks(self, forward_hooks):
        """順伝搬時のhookをすべて登録する
        Parameter
        ---------
        hook :  
        """
        for hook in forward_hooks:
            self._register_forward_hook(hook)
    
    def unregister_all_forward_hooks(self):
        self.forward_hooks = []

    def register_all_backward_hooks(self, backward_hooks):
        """逆伝搬時のhookをすべて登録する
        Parameter
        ---------
        hook : 
        """
        for hook in backward_hooks:
            self._register_backward_hook(hook)
            
    def call_forward_hooks(self, layer_name , input_tensor, output_tensor):
        """順伝搬時のhookをすべて呼び出す(レイヤーの出力に対するhooks)
        Parameter
        ---------
        layer : モデルのレイヤー
        output : レイヤーの出力
        
        """
        for hook in self.forward_hooks:
            hook(layer_name, input_tensor, output_tensor)
    
    def call_backward_hook(self, layer, output):
        """逆伝搬時のhookをすべて呼び出す
        Parameter
        ---------
        layer : モデルのレイヤー
        output : レイヤーの出力
        
        """
        for hook in self.forward_hooks:
            hook(layer, output)
    
    @contextmanager
    def register(self, hooks, enable=True):
        if not enable:
            yield
            return
        
        self.register_all_forward_hooks(hooks)
        try:
            yield
        finally:
            self.unregister_all_forward_hooks()