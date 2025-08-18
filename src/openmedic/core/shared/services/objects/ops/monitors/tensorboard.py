from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List
import logging
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt

from openmedic.core.shared.services.objects.monitor import (
    OpenMedicMonitorOpBase,
    OpenMedicMonitorOpError,
)
from openmedic.core.shared.services.plans.management import OpenMedicOSEnv, OpenMedicPipelineResult
import openmedic.core.shared.services.plans.registry as registry


class TensorBoard(OpenMedicMonitorOpBase):
    def __init__(self, writer: Optional[SummaryWriter], is_activate: bool = True):
        self.writer: Optional[SummaryWriter] = writer
        self.is_activate: bool = is_activate
        self.epoch = 0
        self.history: List[Dict] = []

    @classmethod
    def initialize(cls, is_activate: bool = True):
        writer: Optional[SummaryWriter]
        if is_activate:
            tensorboard_dir: str = os.path.join(
                OpenMedicOSEnv.home,
                OpenMedicPipelineResult.get_current_experiment(),
                "tensorboard"
            )
            os.makedirs(tensorboard_dir, exist_ok=True)
            logging.info(
                f"[TensorBoard][initialize]: Tensorboard files are saved in {tensorboard_dir}"
            )
            writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            writer = None
        return cls(writer, is_activate)

    def execute(self):
        if not self.is_activate or self.writer is None:
            return

        scores = OpenMedicPipelineResult.get_scores() or {}
        result: Dict[str, float] = {}

        if scores.get("train_losses"):
            result["train_loss"] = scores["train_losses"][-1]
        if scores.get("train_metric_scores"):
            result["train_metric"] = scores["train_metric_scores"][-1]
        if scores.get("eval_losses"):
            result["eval_loss"] = scores["eval_losses"][-1]
        if scores.get("eval_metric_scores"):
            result["eval_metric"] = scores["eval_metric_scores"][-1]

        epoch_from_len = max(
            len(scores.get("train_losses", [])),
            len(scores.get("eval_losses", [])),
            self.epoch
        )
        e = epoch_from_len if epoch_from_len > 0 else self.epoch

        self._log_metrics(
            {k: v for k, v in result.items() if isinstance(v, (int, float))},
            epoch=e
        )
        self._save_epoch_result({"epoch": e, **result})
        self._step_epoch()

    def _log_metrics(self, metrics: Dict[str, float], phase: str = "train", epoch: Optional[int] = None):
        if not self.writer:
            return
        e: int = epoch if epoch is not None else self.epoch
        for name, value in metrics.items():
            tag = f"{phase}/{name}"
            self.writer.add_scalar(tag, value, e)

    def _log_learning_rate(self, lr: float, epoch: Optional[int] = None):
        if not self.writer:
            return
        e: int = epoch if epoch is not None else self.epoch
        self.writer.add_scalar("lr", lr, e)

    def _log_histogram(self, name: str, values, epoch: Optional[int] = None):
        if not self.writer:
            return
        e = epoch if epoch is not None else self.epoch
        self.writer.add_histogram(name, values, e)

    def _log_image(self, tag: str, image_tensor, epoch: Optional[int] = None):
        if not self.writer:
            return
        e = epoch if epoch is not None else self.epoch
        grid = torchvision.utils.make_grid(image_tensor)
        self.writer.add_image(tag, grid, e)

    def _log_figure(self, tag: str, figure, epoch: Optional[int] = None):
        if not self.writer:
            return
        e = epoch if epoch is not None else self.epoch
        self.writer.add_figure(tag, figure, e)
        try:
            plt.close(figure)
        except Exception:
            pass

    def _log_confusion_matrix(self, y_true, y_pred, class_names, epoch: Optional[int] = None, normalize: bool = False):
        if not self.writer:
            return

        y_true = np.asarray(list(y_true), dtype=np.int64)
        y_pred = np.asarray(list(y_pred), dtype=np.int64)
        K = len(class_names)
        if K == 0:
            return

        cm = np.zeros((K, K), dtype=np.int64)
        mask = (y_true >= 0) & (y_true < K) & (y_pred >= 0) & (y_pred < K)
        for t, p in zip(y_true[mask], y_pred[mask]):
            cm[t, p] += 1

        cm_display = cm.astype(np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_display = cm_display / row_sums

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cm_display, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        fmt = ".2f" if normalize else "d"
        thresh = cm_display.max() / 2.0 if cm_display.size > 0 else 0.0
        for i in range(K):
            for j in range(K):
                val = cm_display[i, j]
                ax.text(
                    j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                )
        fig.tight_layout()
        self._log_figure("confusion_matrix", fig, epoch)

    def _save_epoch_result(self, result_dict: Dict[str, float]):
        self.history.append(result_dict.copy())

    def _get_all_results(self) -> List[Dict]:
        return self.history

    def _step_epoch(self):
        self.epoch += 1

    def _close(self):
        if self.writer is not None:
            self.writer.close()

def init():
    registry.MonitorRegister.register(monitor_class=TensorBoard)
