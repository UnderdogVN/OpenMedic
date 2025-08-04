from abc import ABC, abstractmethod
from typing import List, Dict
from openmedic.core.shared.services.objects.monitor import TensorBoardLogger

class OpenMedicMonitorOpBase(ABC):
    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def initialize():
        pass

    @abstractmethod
    def execute():
        pass


class OpenMedicMonitorOpError(Exception):
    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicMonitor:
    logger = TensorBoardLogger()
    epoch_history: List[Dict] = []

    def __call__() -> List[object]:
        pass

    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicMonitorOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicMonitorOpBase:
        try:
            return getattr(cls, op_name)
        except AttributeError:
            raise OpenMedicMonitorOpError(f"The {op_name} operation does not exist.")

    @classmethod
    def log_epoch(cls, result: Dict):
        cls.logger.log_metrics(
            {k: v for k, v in result.items() if isinstance(v, (int, float))},
            epoch=result.get("epoch", cls.logger.epoch)
        )
        cls.logger.save_epoch_result(result)
        cls.logger.step_epoch()
        cls.epoch_history.append(result)

    @classmethod
    def close(cls):
        cls.logger.close()

    @classmethod
    def get_all_results(cls) -> List[Dict]:
        return cls.epoch_history
