import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

import openmedic.core.shared.services.plans.registry as registry
import openmedic.core.shared.services.utils as utils
from openmedic.core.shared.services.objects.model import OpenMedicModelBase
from openmedic.core.shared.services.objects.monitor import (
    OpenMedicMonitorOpBase,
    OpenMedicMonitorOpError,
)


@dataclass
class CheckPoint(OpenMedicMonitorOpBase):
    def __init__(
        self,
        model_path: str,
        is_replace: bool = False,
        early_stopping: dict = {},
        save_best: dict = {},
    ):
        self.model_path: str = model_path
        self.is_replace: bool = is_replace
        self.early_stopping: dict = early_stopping
        self.best_score: Optional[float] = None
        self.save_best: dict = save_best
        self._count: int = 0

    @classmethod
    def initialize(
        cls,
        model_path: str = "",
        is_replace: bool = False,
        early_stopping: dict = {},
        save_best: dict = {},
        *args,
        **kwargs,
    ):
        if not model_path:
            raise OpenMedicMonitorOpError("`model_path` can not found in config.")

        if not model_path.endswith(".pth"):
            raise OpenMedicMonitorOpError(
                "Can not save model with different format `.pth`",
            )

        if os.path.exists(model_path) and not is_replace:
            raise OpenMedicMonitorOpError(
                f"The {model_path} is already exist! If you want to replace that file, please set `is_replace` is true",
            )

        if save_best:
            patience: Optional[int] = save_best.get("patience", None)
            if patience == 0:
                raise OpenMedicMonitorOpError(
                    "Can not set `patience` equal to 0. If you do not want apply earl stopping, please remove `patience`.",
                )

        return cls(model_path, is_replace, early_stopping, save_best)

    def _get_target_score_name(self) -> str:
        target_dataset: str = self.save_best.get("target_data", None)
        assert isinstance(
            target_dataset,
            str,
        ), "`target_dataset` parameter does not exist in `save_best`"
        target_score: str = self.save_best.get("target_score", None)
        assert isinstance(
            target_score,
            str,
        ), "`target_dataset` parameter does not exist in `save_best`"

        return f"{target_dataset}_{target_score}"

    def _save(self, model):
        pass

    def execute(self, model: OpenMedicModelBase, **kwargs):
        is_save: bool = True
        if self.save_best:
            score_name: str = self._get_target_score_name()
            score: Optional[float] = kwargs.get(score_name, None)
            assert isinstance(
                score,
                float,
            ), f"{score_name} can not be found as input parameters in `CheckPoint.execute()`"

            patience: Optional[int] = self.save_best.get("patience", None)
            compare_operator = "<" if self.save_best["is_better"] == "increase" else ">"

            if self.best_score:
                flag: bool = eval(f"{self.best_score} {compare_operator} {score}")
                if flag:
                    logging.info(
                        f"[CheckPoint][execute]: Found best score {score} from {self.best_score}",
                    )
                    self.best_score = score
                    # Reset `_count`
                    self._count = 0
                else:
                    self._count += 1
                    is_save = False
            else:
                logging.info(f"[CheckPoint][execute]: Save best score to {score}")
                self.best_score = score

        if is_save:
            torch.save(model.state_dict(), self.model_path)
            logging.info(f"[CheckPoint][execute]: Saved model to {self.model_path}.")

        if patience:
            if self._count >= patience:
                raise utils.BreakLoop(
                    f"[CheckPoint][Exception]: The score has stopped improving after {patience} times",
                )


def init():
    registry.MonitorRegister.register(monitor_class=CheckPoint)
