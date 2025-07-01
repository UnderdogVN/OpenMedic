from abc import ABC, abstractmethod
from typing import List

import openmedic.core.shared.services.utils as utils
import openmedic.core.shared.services.objects.model as model
import openmedic.core.shared.services.objects.transform as transform
import openmedic.core.shared.services.objects.metric as metric
import openmedic.core.shared.services.objects.loss_function as lf
import openmedic.core.shared.services.objects.monitor as monitor


class BaseRegister(ABC):
    """The register abstract class.

    Usage:
    ------
    ```
    ### In modules.module1.py ###
    class ModuleClass:
        pass

    # Registers `ModuleClass` into `BaseRegister`.
    BaseRegister.register(module=ModuleClass)

    ### In registry.py ###
    # Imports all modules that are listed in `__MODULES` and registered by `register` method.
    # Note: All modules need to be located corresponding to `__TEMPLATE`.
    BaseRegister.import_modules()
    ```

    """
    __TEMPLATE: str
    __MODULES: List[str]

    @abstractmethod
    def import_modules(cls):
        """Imports class modules that are defined in `__MODULES`
            and are located in `__TEMPLATE`
        """
        pass

    @abstractmethod
    def register(*args, **kwargs):
        """Registers class module"""
        pass


class ModelRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.model_zoo.{module}"
    __MODULES: List[str] = [
        "unet"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, model_class: model.OpenMedicModelBase, model_name: str=''):
        if not model_name:
            model_name = model_class.get_name()
        model.OpenMedicModel.add_model(model_name=model_name, model_class=model_class)


class TransformRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.transformers.{module}"
    __MODULES: List[str] = [
        "resize",
        "flip"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            model_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            model_interface.init()

    @classmethod
    def register(cls, transform_class: transform.OpenMedicTransformOpBase, transfrom_name: str=''):
        if not transfrom_name:
            transfrom_name = transform_class.get_name()
        transform.OpenMedicTransform.add_op(op_name=transfrom_name, op_class=transform_class)


class MetricRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.metrics.{module}"
    __MODULES: List[str] = [
        "dice_score"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, metric_class: metric.OpenMedicMetricOpBase, metric_name: str=''):
        if not metric_name:
            metric_name = metric_class.get_name()
        metric.OpenMedicMetric.add_op(op_name=metric_name, op_class=metric_class)


class LossRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.losses.{module}"
    __MODULES: List[str] = [
        "dice_loss"
    ]

    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, loss_class: lf.OpenMedicLossOpBase, loss_name: str=''):
        if not loss_name:
            loss_name = loss_class.get_name()
        lf.OpenMedicLossFunction.add_op(op_name=loss_name, op_class=loss_class)


class MonitorRegister(BaseRegister):
    __TEMPLATE: str = "openmedic.core.shared.services.objects.ops.monitors.{module}"
    __MODULES: List[str] = [
        "checkpoint"
    ]
    @classmethod
    def import_modules(cls):
        for module in cls.__MODULES:
            module_interface: utils.ModuleInterface = utils.import_module(module_name=cls.__TEMPLATE.format(module=module))
            module_interface.init()

    @classmethod
    def register(cls, monitor_class: monitor.OpenMedicMonitorOpBase, monitor_name: str=''):
        if not monitor_name:
            monitor_name = monitor_class.get_name()
        monitor.OpenMedicMonitor.add_op(op_name=monitor_name, op_class=monitor_class)


class OpenMedicRegsiter:
    model_register: ModelRegister = ModelRegister
    transform_register: TransformRegister = TransformRegister
    metric_register: MetricRegister = MetricRegister
    loss_register: LossRegister = LossRegister
    monitor_register: MonitorRegister = MonitorRegister

    @classmethod
    def init(cls):
        cls.model_register.import_modules()
        cls.transform_register.import_modules()
        cls.metric_register.import_modules()
        cls.loss_register.import_modules()
        cls.monitor_register.import_modules()