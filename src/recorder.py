# trainer:
#   recorder:
#     enable: true
#     save_forward: true
#     save_backward: true
#     max_samples: 10
    
class Recorder:
    """
    順伝搬, 逆伝搬用のデータを記録する
    """
    def __init__(self):
        pass
    
    def forward_hook(self, layer, out):
        print(f"順伝搬{layer, out.shape}")

    def backward_hook(self, layer, dout):
        print(f"逆伝搬{layer, dout.shape}")