from abc import ABC, abstractmethod
from typing import Type, Any

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
    @classmethod
    def add_op(cls, op_name: str, op_class: Type[Any]) -> None:
        setattr(cls, op_name, op_class)
    @classmethod
    def get_op(cls, op_name: str) -> Type[Any]:
        try:
            return getattr(cls, op_name)
        except AttributeError:
            raise OpenMedicMonitorOpError(f"The {op_name} operation does not exist.")
