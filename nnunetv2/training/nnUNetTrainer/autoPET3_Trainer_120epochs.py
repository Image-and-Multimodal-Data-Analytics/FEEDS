import os

import torch

from nnunetv2.training.nnUNetTrainer import autoPET3_Trainer


class autoPET3_Trainer_120epochs(autoPET3_Trainer):
    os.environ["STEM"] = "autoPET"
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 120
        self.initial_lr = 1e-3