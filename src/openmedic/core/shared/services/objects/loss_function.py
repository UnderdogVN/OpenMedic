import torch
import torch.nn as nn

from openmedic.core.shared.services.config import ConfigReader


class OpenMedicLossOpBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.__name__


class OpenMedicLossOpError(Exception):
    """Custom exception"""

    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicLossFunction:
    @classmethod
    def _preprocess_torch(cls, **kwargs) -> dict:
        # Process `weight` parameter from list to torch.Tensor
        if kwargs.get("weight", None):
            device: str = "cpu"
            if ConfigReader.get_field(name="pipeline").get("is_gpu", False):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kwargs["weight"] = torch.tensor(kwargs["weight"], device=device)

        return kwargs

    @classmethod
    def get_torch_loss_function(cls, name: str, **kwargs) -> nn.Module:
        kwargs: dict = cls._preprocess_torch(**kwargs)
        return getattr(nn, name)(**kwargs)

    @classmethod
    def get_custom_loss_function(cls, name: str, **kwargs) -> nn.Module:
        return getattr(cls, name)(**kwargs)

    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicLossOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicLossOpBase:
        error_msg: str = ""
        try:
            return getattr(cls, op_name)
        except AttributeError:
            error_msg = f"The {op_name} operation does not exist."
            raise OpenMedicLossOpError(message=error_msg)
