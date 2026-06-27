from sequential import Sequential
from model_builder import ModelBuilder
from utils import load_yaml
from pathlib import Path

yaml_file = Path(r"config/VGG16.yaml")
config = load_yaml(yaml_file)
mb = ModelBuilder(config)

layers = mb.build()

for layer in layers:
    print(layer)