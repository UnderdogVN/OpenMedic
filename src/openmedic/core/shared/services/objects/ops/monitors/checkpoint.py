from dataclasses import dataclass
import os
import torch
import logging
from typing import Optional, List

from openmedic.core.shared.services.objects.monitor import OpenMedicMonitorOpBase, OpenMedicMonitorOpError
import openmedic.core.shared.services.plans.registry as registry
from openmedic.core.shared.services.objects.model import OpenMedicModelBase
import openmedic.core.shared.services.utils as utils
from openmedic.core.shared.services.plans.management import OpenMedicPipelineResult, OpenMedicOSEnv


@dataclass
class CheckPoint(OpenMedicMonitorOpBase):
    def __init__(self, model_dir: str, model_file: str, is_replace: bool=False, save_best: dict={}, target_score: str='', patience: Optional[None]=None):
        self.model_dir: str = model_dir
        self.model_file: str = model_file
        self.is_replace: bool = is_replace
        self.best_score: Optional[float] = None
        self.save_best: dict = save_best
        self._count: int = 0
        self.target_score: str = target_score
        self.patience: Optional[int] = patience

    @classmethod
    def initialize(
        cls,
        model_dir: str='',
        model_file: str='',
        is_replace: bool=False,
        save_best: dict={},
        *args,
        **kwargs
    ):
        if bool(model_dir) != bool(model_file):
            raise OpenMedicMonitorOpError("Need to declare `model_dir` and `model_file` at the same time.")

        if model_file and not model_file.endswith(".pth"):
            raise OpenMedicMonitorOpError("Can not save model with different format `.pth`")

        if model_dir and model_file:
            model_path: str = os.path.join(model_dir, model_file)
            if os.path.exists(model_path) and not is_replace:
                raise OpenMedicMonitorOpError(f"The {model_path} is already exist! If you want to replace that file, please set `is_replace` is true")
        else:
            model_dir = os.path.join(OpenMedicOSEnv.home, OpenMedicPipelineResult.get_current_experiment())
            model_file = "model.pth"

        os.makedirs(name=model_dir, exist_ok=True)
        target_score: str = ''
        patience: Optional[int] = None
        if save_best:
            patience = save_best.get("patience", None)
            if patience == 0:
                raise OpenMedicMonitorOpError("Can not set `patience` equal to 0. If you do not want apply earl stopping, please removes `patience`.")

            target_score: str = save_best.get("target_score", '')
            SCORE_SUPPORTS: list = [
                "train_losses",
                "train_metric_scores",
                "eval_losses",
                "eval_metric_scores"
            ]
            if target_score not in SCORE_SUPPORTS:
                if target_score == '':
                    raise OpenMedicMonitorOpError("Needs to set `target_score` parameter in `save_best`.")
                else:
                    raise OpenMedicMonitorOpError(f"Expects value {','.join(SCORE_SUPPORTS)} in `target_score`.")
        return cls(model_dir, model_file, is_replace, save_best, target_score, patience)

    def _get_target_score_name(self) -> str:
        target_dataset: str = self.save_best.get("target_data", None)
        assert isinstance(target_dataset, str), "`target_dataset` parameter does not exist in `save_best`"
        target_score: str = self.save_best.get("target_score", None)
        assert isinstance(target_score, str), "`target_dataset` parameter does not exist in `save_best`"

        return f"{target_dataset}_{target_score}"

    def _save(self, model):
        pass

    def execute(self, **kwargs):
        is_save_model: bool = True
        if self.save_best:
            scores: List[float] = OpenMedicPipelineResult.get_result(attr_name=self.target_score)
            latest_score: float = scores[-1]

            compare_operator = '>' if self.target_score.endswith("losses") else '<'
            if self.best_score:
                if eval(f"{self.best_score} {compare_operator} {latest_score}"):
                    logging.info(f"[CheckPoint][execute]: Found best score {latest_score} from {self.best_score}")
                    self.best_score = latest_score
                    # Reset `_count`
                    self._count = 0
                else:
                    self._count += 1
                    is_save_model = False
            else:
                logging.info(f"[CheckPoint][execute]: Save best score to {latest_score}")
                self.best_score = latest_score


        metadata: dict = OpenMedicPipelineResult.get_metadata()
        metadata_path: str = os.path.join(self.model_dir, "metadata.yml")
        scores_data: dict = OpenMedicPipelineResult.get_scores()
        scores_data_path: str = os.path.join(self.model_dir, "checkpoint.json")
        utils.save_as_yml(data=metadata, file_path=metadata_path, if_exist="overwrite")
        utils.save_as_json(data=scores_data, file_path=scores_data_path, if_exist="overwrite")

        if is_save_model:
            model: OpenMedicModelBase = OpenMedicPipelineResult.get_model()
            model_path: str = os.path.join(self.model_dir, self.model_file)
            torch.save(model.state_dict(), model_path)

        if self.patience:
            if self._count >= self.patience:
                raise utils.BreakLoop(f"[CheckPoint][Exception]: The score has stopped improving after {self.patience} times")


def init():
    registry.MonitorRegister.register(monitor_class=CheckPoint)