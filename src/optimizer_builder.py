from optimizer_registry import get_optimizer_builder

class OptimizerBuilder:
    def __init__(self, config):
        self.config = config
    
    def build(self):
        optimizer_config = self.config["optimizer"]

        optimizer_builder = get_optimizer_builder(optimizer_config["type"])
        return optimizer_builder(optimizer_config)