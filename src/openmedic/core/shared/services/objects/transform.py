from typing import List
from abc import ABC, abstractmethod
import numpy as np


class OpenMedicTransformOpBase(ABC):
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
    def execute(image: np.ndarray, gt: np.ndarray, *args, **kwargs) -> List[np.ndarray]:
        """Execute transform operator.

        Input:
        ------
            image: np.ndarray - Image.
            gt: np.ndarray - Ground truth image.

        Output:
        -------
            image, gt: List[np.ndarray] - Return transformed image and ground truth.
        """
        pass


class OpenMedicTransformOpError(Exception):
    """Custom exception."""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicTransform:
    def __call__() -> List[object]:
        pass

    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicTransformOpBase):
        """Add operator `OpenMedicTransformOpBase`.
        Input:
        ------
            op_name: str - Operator name.
            op_class: OpenMedicTransformOpBase - Transform operator.
        """
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicTransformOpBase:
        """Get operator `OpenMedicTransformOpBase`.
        Input:
        ------
            op_name: str - Operator name.

        Output:
        ------
            operator: OpenMedicTransformOpBase.
        """
        error_msg: str = ''
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg=f"The {op_name} operation does not exist."
            raise OpenMedicTransformOpError(message=error_msg)