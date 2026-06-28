from sequential import Sequential
from model_builder import ModelBuilder
from utils import load_yaml
from pathlib import Path
import torch

# class VGG16:
#     def __init__(self, yaml_file):        
#         yaml_file = Path(r"config/VGG16.yaml")
#         config = load_yaml(yaml_file)
#         mb = ModelBuilder(config)
#         layers = mb.build()
        
#         self.sequential = Sequential(layers)
#         self.device =None
#         self.params = self.sequential.params
#         self.grads = self.sequential.grads

#     def forward(self, x):
#         self.sequential.forward(x)
    
#     def backward(self, dout):
#         self.sequential.backward(dout)

yaml_file = Path(r"config/VGG16.yaml")
config = load_yaml(yaml_file)
mb = ModelBuilder(config)
layers = mb.build()
model = Sequential(layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for layer in layers:
#     print(layer)

# print(len(layers))

# print(model)

# x = torch.randn((2, 3, 224, 224))
# out = model.forward(x)

# print(out.device)