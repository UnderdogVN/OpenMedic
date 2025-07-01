from abc import ABC, abstractmethod
from typing import List


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
    """Custom exception"""

    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicMonitor:
    def __call__() -> List[object]:
        pass

    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicMonitorOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicMonitorOpBase:
        error_msg: str = ""
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg = f"The {op_name} operation does not exist."
            raise OpenMedicMonitorOpError(message=error_msg)
