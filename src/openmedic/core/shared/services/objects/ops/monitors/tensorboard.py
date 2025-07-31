
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


class TensorBoardLogger:
    def __init__(self, log_dir: str = "runs/openmedic"):
        self.writer = SummaryWriter(log_dir)
        self.epoch = 0
        self.history: List[Dict] = []

    def log_metrics(self, metrics: Dict[str, float], phase: str = "train", epoch: Optional[int] = None):
        e = epoch if epoch is not None else self.epoch
        for name, value in metrics.items():
            tag = f"{phase}/{name}"
            self.writer.add_scalar(tag, value, e)

    def log_learning_rate(self, lr: float, epoch: Optional[int] = None):
        e = epoch if epoch is not None else self.epoch
        self.writer.add_scalar("lr", lr, e)

    def log_histogram(self, name: str, values, epoch: Optional[int] = None):
        e = epoch if epoch is not None else self.epoch
        self.writer.add_histogram(name, values, e)

    def log_image(self, tag: str, image_tensor, epoch: Optional[int] = None):
        e = epoch if epoch is not None else self.epoch
        grid = torchvision.utils.make_grid(image_tensor)
        self.writer.add_image(tag, grid, e)

    def log_figure(self, tag: str, figure, epoch: Optional[int] = None):
        e = epoch if epoch is not None else self.epoch
        self.writer.add_figure(tag, figure, e)

    def log_confusion_matrix(self, y_true, y_pred, class_names, epoch: Optional[int] = None):
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True")
        plt.xlabel("Pred")
        self.log_figure("confusion_matrix", fig, epoch)

    def save_epoch_result(self, result_dict: Dict[str, float]):
        self.history.append(result_dict.copy())

    def get_all_results(self) -> List[Dict[str, float]]:
        return self.history

    def step_epoch(self):
        self.epoch += 1

    def close(self):
        self.writer.close()
