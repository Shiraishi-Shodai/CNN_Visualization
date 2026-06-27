from model_builder import ModelBuilder
from utils import load_yaml
from pathlib import Path
from sequential import Sequential
import torch

yaml_file = Path(r"config/simple_dense2.yaml")

config = load_yaml(yaml_file)
model_builder = ModelBuilder(config)
layers = model_builder.build()

model = Sequential(layers)

x = torch.randint(1, 100, (3, 2), dtype=torch.float32)
out = model.forward(x)

dout = model.backward(out)
print(dout)
