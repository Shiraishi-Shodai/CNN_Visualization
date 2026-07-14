from custom_dataclasses import EpochMetrics, EvaluationMetrics
import torch

class History:

    def __init__(self, class_num):   
        self.train: list[EpochMetrics] = []
        self.valid: list[EpochMetrics] = []
        self.test: list[EpochMetrics] = []
        
        self.train_eval: EvaluationMetrics | None = EvaluationMetrics(
                                   torch.zeros(class_num, class_num, dtype=torch.int16),
                                   torch.zeros(class_num),
                                   )
        
        self.valid_eval: EvaluationMetrics | None = EvaluationMetrics(
                                   torch.zeros(class_num, class_num, dtype=torch.int16),
                                   torch.zeros(class_num),
                                   )
        self.test_eval: EvaluationMetrics | None = EvaluationMetrics(
                                   torch.zeros(class_num, class_num, dtype=torch.int16),
                                   torch.zeros(class_num),
                                   )
        
    def get_metrics(self, mode):
        """使用するEvaluationMetricsを返す
        """
        assert mode in ["train", "valid", "test"], "train or valid or testを入力してください"
        
        match mode:
            case "train":
                return self.train
            case "valid": 
                return self.valid
            case "test":
                return self.test
            
    def get_evaluation_metrics(self, mode):
        """使用するEvaluationMetricsを返す
        """
        assert mode in ["train", "valid", "test"], "train or valid or testを入力してください"
        
        match mode:
            case "train":
                return self.train_eval
            case "valid": 
                return self.valid_eval
            case "test":
                return self.test_eval