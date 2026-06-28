import torch
from sequential import Sequential
from model_builder import ModelBuilder
from utils import load_yaml
from pathlib import Path
import torch
import math
from layers import SoftmaxWithLoss
from optimizer import SGD
import copy

print(torch.cuda.is_available())
print(f"認識されているGPUの数 {torch.cuda.device_count()}")
print(f"デバイスID {torch.cuda.get_device_name()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"利用可能なデバイス {device}")

yaml_file = Path(r"config/VGG16.yaml")
config = load_yaml(yaml_file)
mb = ModelBuilder(config)
layers = mb.build()
# model = Sequential(layers)

"""パラメータdeviceチェック"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"=====deviceセット前=====")
# for layer in model.layers:
#     for i in range(len(layer.params)):
#         print(f"params : {layer.params[i].device}")
#         print(f"grads : {layer.grads[i].device}")

# model.to(device)

# print(f"=====deviceセット後=====")
# for layer in model.layers:
#     for i in range(len(layer.params)):
#         print(f"params : {layer.params[i].device}")
#         print(f"grads : {layer.grads[i].device}")


# x = torch.randn((2, 3, 224, 224)).to(device)
# x = torch.randn((2, 3, 5, 5)).to(device)

""""順伝搬と逆伝搬 deviceチェック"""
# out = model.forward(x)
# print(f"順伝搬出力結果デバイス {out.device}, shape { out.shape}")

# dout = model.backward(out)
# print(f"逆伝搬出力結果デバイス {dout.device}, shape {dout.shape}")

"""cpu gpuチェック"""

def CPUGPUvalueCheck(cpu_data, gpu_data):
    """CPUとGPUの場合で、出力される値が一致することをチェック
    """
    
    allclose_check = torch.allclose(
        cpu_data,
        gpu_data,
        atol=1e-6
    )

    error_check = torch.max(torch.abs(cpu_data - gpu_data.cpu()))
    
    return allclose_check, error_check

# x = torch.randn((2, 3, 224, 224))
# device = torch.device("cpu")
# model.to(device)
# cpu_out = model.forward(x.to(device))
# cpu_dout = model.backward(cpu_out)

# device = torch.device("cuda")
# model.to(device)
# gpu_out = model.forward(x.to(device))
# gpu_dout = model.backward(gpu_out)

# # 順伝搬のチェック
# print(CPUGPUvalueCheck(cpu_out, gpu_out.cpu()))
# print(CPUGPUvalueCheck(cpu_dout, gpu_dout.cpu()))

"""学習ループのチェック"""
model_cpu = Sequential(copy.deepcopy(layers)) # 深いコピー
model_cpu.to("cpu")
model_gpu = Sequential(copy.deepcopy(layers)) # 深いコピー
model_gpu.to("cuda")

# for cpu, gpu in zip(model_cpu.layers, model_gpu.layers):
#     for cpu_param, gpu_param in zip(cpu.params, gpu.params):
#         print(cpu_param.device, gpu_param.device)

x_cpu = torch.randn((3, 3, 224, 224)).to("cpu")
t_cpu = torch.tensor([8, 1, 4]).to("cpu")

x_gpu = torch.randn((3, 3, 224, 224)).to("cuda")
t_cpu = torch.tensor([8, 1, 4]).to("cuda")

cpu_criterion = SoftmaxWithLoss()
gpu_criterion = SoftmaxWithLoss()

optimizer = SGD(lr=0.01)

for step in range(3):
    cpu_out = model_cpu.forward(x_cpu)
    cpu_loss = cpu_criterion.forward(cpu_out, t_cpu)
    cpu_dout = cpu_criterion.backward()
    cpu_dx = model_cpu.backward(cpu_dout)
    optimizer.update(model_cpu.params, model_cpu.grads)
    
    print(model_cpu.params)
    
## flattenを2次元に直す！！！！