import torch.optim as optim
import torch.nn as nn


class OpenMedicOptimizerOpBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.__name__


class OpenMedicOptimizerOpError(Exception):
    """Custom exception"""
    def __init__(self, message: str=''):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicOptimizer:
    @staticmethod
    def get_torch_optimization(name: str, **kwargs) -> optim.Optimizer:
        error_msg: str = ''
        try:
            return getattr(optim, name)(**kwargs)
        except AttributeError:
            error_msg=f"The {name} operation does not exist."
            raise OpenMedicOptimizerOpError(message=error_msg)