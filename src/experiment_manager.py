from datetime import datetime
from pathlib import Path
import shutil
from custom_dataclasses import ExperimentSubDirs
from checkpoint_manager import CheckPointManager
import torch
from dataclasses import fields

class ExperimentManager:
    def __init__(self):
        self.experiment_dir = fr"notes/{datetime.now().strftime("%Y%m%d_%H%M%S")}"
        self.experiment_sub_dir = ExperimentSubDirs(
            models=fr"{self.experiment_dir}/models",
            config=fr"{self.experiment_dir}/config",
            imgs=fr"{self.experiment_dir}/imgs",
            
        )
    
    def create(self):        

        for sub_dir in fields(self.experiment_sub_dir):
            # configの場合はフォルダごとコピーしてくる
            if sub_dir.name == "config":
                shutil.copytree(
                    "config",
                    getattr(self.experiment_sub_dir, sub_dir.name),
                    dirs_exist_ok=True
                )
                continue
            
            # config以外はフォルダだけ新規作成
            Path(getattr(self.experiment_sub_dir, sub_dir.name)).mkdir(parents=True, exist_ok=True)

