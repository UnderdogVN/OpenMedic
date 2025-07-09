import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.objects.metric as metric
import openmedic.core.shared.services.objects.monitor as mt
import openmedic.core.shared.services.objects.optimization as optimization
from openmedic.core.shared.services.config import ConfigReader
from openmedic.core.shared.services.objects.model import (
    OpenMedicModel,
    OpenMedicModelBase,
)


class OpenMedicTrainerException(Exception):
    """Customizes exception"""

    def __init__(self, message: str = "An error occurred in OpenMedicTrainerException"):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicTrainer:
    """OpenMedicTrainer manages OpenMedic objects to serve training process."""

    ### CLASS METHODS ###
    @classmethod
    def initialize_with_config(cls):
        """Initializes object.
        Note: The ConfigReader need to be initialized before.
        """
        model_info: dict = ConfigReader.get_field(name="model")
        optim_info: dict = ConfigReader.get_field(name="optimization")
        loss_info: dict = ConfigReader.get_field(name="loss_function")
        metric_info: dict = ConfigReader.get_field(name="metric")
        monitor_info: Optional[dict] = ConfigReader.get_field(name="monitor")

        model: OpenMedicModelBase = cls._get_model(model_info=model_info)
        optimizer: optim.Optimizer = cls._get_optimizer(
            model=model,
            optim_info=optim_info,
        )
        loss_function: nn.modules = cls._get_loss_function(loss_info=loss_info)
        metric_op: metric.OpenMedicMetricOpBase = cls._get_metric(
            metric_info=metric_info,
        )
        monitor_map_ops: Dict[str, mt.OpenMedicMonitorOpBase] = (
            cls._get_monitor_map_ops(monitor_info=monitor_info)
        )
        return cls(model, optimizer, loss_function, metric_op, monitor_map_ops)

    @classmethod
    def _get_model(cls, model_info: dict) -> OpenMedicModelBase:
        """Gets OpenMedic model.

        Input:
        ------
            model_info: dict - The model information.

        Output:
        -------
            OpenMedicModelBase - Returns OpenMedic model.
        """
        model_name: str = model_info["name"]
        model_params: dict = model_info["params"]
        model_checkpoint: str = model_info.get("model_checkpoint", "")
        model: OpenMedicModelBase = OpenMedicModel.get_model(model_name=model_name)(
            **model_params,
        )
        if model_checkpoint:
            # TODO: Check related or absolute path
            model.load_state_dict(torch.load(model_checkpoint))

        return model

    @classmethod
    def _get_optimizer(
        cls,
        model: OpenMedicModelBase,
        optim_info: dict,
    ) -> optim.Optimizer:
        """Gets Torch Optimizer
        TODO: Need to implement to logic get custom Optimizer.

        Input:
        -----
            model: OpenMedicModelBase - The OpenMedic model.
            optim_info: dict - The information of optimization.

        Output:
        ------
            optim.Optimizer - Returns Torch Optimizier.
        """
        optim_name: str = optim_info["name"]
        optim_params: str = optim_info["params"]
        params: dict = {
            "params": model.parameters(),
        }
        params.update(optim_params)
        return optimization.OpenMedicOptimizer.get_torch_optimization(
            name=optim_name,
            **params,
        )

    @classmethod
    def _get_loss_function(cls, loss_info: dict) -> nn.Module:
        """Gets the loss function.

        Input:
        -----
            loss_info: dict - The information of loss function.

        Output:
        ------
            nn.Moduler - Returns the loss function.
        """
        loss_name: str = loss_info["name"]
        loss_params: dict = loss_info["params"]
        loss_type: str = loss_info["type"]
        if loss_type == "torch":
            return lf.OpenMedicLossFunction.get_torch_loss_function(
                name=loss_name,
                **loss_params,
            )
        elif loss_type == "custom":
            return lf.OpenMedicLossFunction.get_custom_loss_function(
                name=loss_name,
                **loss_params,
            )
        else:
            raise OpenMedicTrainerException(
                f"`loss_type` ({loss_type}) is not support.",
            )

    @classmethod
    def _get_monitor_map_ops(
        cls,
        monitor_info: Optional[dict],
    ) -> Dict[str, mt.OpenMedicMonitorOpBase]:
        """Gets the monitor operators

        Input:
        -----
            monitor_info: dict - The information of monitor.

        Output:
        ------
            Dict[str, mt.OpenMedicMonitorOpBase] - The dictionary of monitor operators.
        """
        if not monitor_info:
            logging.warning(
                f"[{cls.__name__}][_get_monitor_manager]: You did not configure `monitor`",
            )
            return {}

        monitor_map_ops: Dict[str, mt.OpenMedicMonitorOpBase] = {}
        for op_name, params in monitor_info.items():
            if not isinstance(params, dict):
                raise OpenMedicTrainerException(
                    f"The `{op_name}` expects its parameters as dictionay.",
                )

            monitor_map_ops.update(
                {
                    op_name: mt.OpenMedicMonitor.get_op(op_name=op_name).initialize(
                        **params,
                    ),
                },
            )

        return monitor_map_ops

    @classmethod
    def _get_metric(cls, metric_info: dict) -> metric.OpenMedicMetricOpBase:
        """Gets the metrics.

        Input:
        -----
            loss_info: dict - The information of metric.

        Output:
        ------
            metric.OpenMedicMetricOpBase - Return OpenMedic metric.
        """
        metric_name: str = metric_info["name"]
        metric_params: dict = metric_info["params"]

        return metric.OpenMedicMetric.get_op(op_name=metric_name)(**metric_params)

    ### OBJECT METHODS ###
    def __init__(
        self,
        model: OpenMedicModelBase,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
        metric_op: metric.OpenMedicMetricOpBase,
        monitor_map_ops: Dict[str, mt.OpenMedicMonitorOpBase],
    ):
        """Note: Recommend to initalize object by `initialize_with_config` method.

        Input:
        ------
            model: OpenMedicModelBase - The OpenMedic Model.
            optimizer: optim.Optimizer - The Torch optimizer.
            loss_function: nn.Module - The OpenMedic loss function.
            metric_op: metric.OpenMedicMetricOpBase - The OpenMedic metric operator.
            monitor_map_ops: Dict[str, mt.OpenMedicMonitorOpBase] - The dictonary of OpenMedic monitor operators.
        """
        self.model: OpenMedicModelBase = model
        self.optimizer: optim.Optimizer = optimizer
        self.loss_function: nn.Module = loss_function
        self.metric_op: metric.OpenMedicMetricOpBase = metric_op
        self.monitor_map_ops: Dict[str, mt.OpenMedicMonitorOpBase] = monitor_map_ops

    def get_model(self) -> OpenMedicModelBase:
        return self.model

    def feedforward(self, images: torch.Tensor, gts: torch.Tensor) -> list:
        """Executes feed forward process.

        Input:
        -----
            images: torch.Tensor - Batch of images.
            gts: torch.Tensor - Batch of ground truths.
        """
        gts_pred: torch.Tensor = self.model(images)
        losses: torch.Tensor = self.loss_function(gts_pred, gts)
        res: dict = self.metric_op.execute(gts_pred=gts_pred, gts=gts)
        metric_score = res["mean"]

        return losses, metric_score

    def get_object(self, names: List[str]) -> list:
        return [getattr(self, name) for name in names]

    def _execute_check_point(self, **kwargs):
        import openmedic.core.shared.services.objects.ops.monitors.checkpoint as checkpoint

        checkpoint_handler: Optional[checkpoint.CheckPoint] = self.monitor_map_ops.get(
            "CheckPoint",
            None,
        )
        if not checkpoint_handler:
            return

        checkpoint_handler.execute(**kwargs)

    def execute_monitor(self, **kwargs):
        for monitor_op in self.monitor_map_ops.values():
            monitor_op.execute(**kwargs)
