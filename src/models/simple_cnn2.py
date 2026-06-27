from model_builder import ModelBuilder
from pathlib import Path
from utils import load_yaml
from sequential import Sequential
import torch

yaml_file = Path("config/simple_cnn2.yaml")
config = load_yaml(yaml_file)
model_builder = ModelBuilder(config)

layers = model_builder.build()

# for layer in layers:
#     print(layer)
#     for i in range(len(layer.params)):
#         print(layer.params[i].shape, layer.grads[i].shape)

x = torch.randn((2, 3, 32, 32))
model = Sequential(layers)

out = model.forward(x)
print(out.shape)

dout = model.backward(out)
print(dout.shape)