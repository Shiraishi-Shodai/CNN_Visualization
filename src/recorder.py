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
    
    def forward_hook(self, layer, output):
        print(layer, output.shape)