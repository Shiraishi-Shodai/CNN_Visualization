import torch

class CheckPointManager:
    def __init__(self):
        pass
    
    def save(self, params, file_path="model.pth"):
        torch.save(params, file_path)
    
    def load(self, file_path):
        return torch.load(file_path)