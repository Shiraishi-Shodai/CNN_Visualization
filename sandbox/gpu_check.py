import torch
from sequential import Sequential
from model_builder import ModelBuilder
from utils import load_yaml
from pathlib import Path
import torch
import math

print(torch.cuda.is_available())
print(f"認識されているGPUの数 {torch.cuda.device_count()}")
print(f"デバイスID {torch.cuda.get_device_name()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"利用可能なデバイス {device}")

yaml_file = Path(r"config/VGG16.yaml")
config = load_yaml(yaml_file)
mb = ModelBuilder(config)
layers = mb.build()
model = Sequential(layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=====deviceセット前=====")
for layer in model.layers:
    for i in range(len(layer.params)):
        print(f"params : {layer.params[i].device}")
        print(f"grads : {layer.grads[i].device}")

model.to(device)

print(f"=====deviceセット後=====")
for layer in model.layers:
    for i in range(len(layer.params)):
        print(f"params : {layer.params[i].device}")
        print(f"grads : {layer.grads[i].device}")


x = torch.randn((2, 3, 224, 224)).to(device)
# x = torch.randn((2, 3, 5, 5)).to(device)

out = model.forward(x)
print(f"順伝搬出力結果デバイス {out.device}, shape { out.shape}")

dout = model.backward(out)
print(f"逆伝搬出力結果デバイス {dout.device}, shape {dout.shape}")
