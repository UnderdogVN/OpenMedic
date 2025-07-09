from abc import ABC, abstractmethod


class OpenMedicMetricOpBase(ABC):
    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(self) -> dict:
        pass


class OpenMedicOpError(Exception):
    """Custom exception"""

    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicMetric:
    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicMetricOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicMetricOpBase:
        error_msg: str = ""
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg = f"The {op_name} operation does not exist."
            raise OpenMedicOpError(message=error_msg)
