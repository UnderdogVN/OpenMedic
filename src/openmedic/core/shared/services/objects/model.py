import torch.nn as nn


class OpenMedicModelError(Exception):
    """Custom exception"""

    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.__name__


class OpenMedicModel:
    @classmethod
    def add_model(cls, model_name: str, model_class: OpenMedicModelBase):
        setattr(cls, model_name, model_class)

    @classmethod
    def get_model(cls, model_name) -> OpenMedicModelBase:
        error_msg: str = ""
        try:
            return getattr(cls, model_name)
        except AttributeError:
            error_msg = f"The {model_name} model does not exist."
            raise OpenMedicModelError(message=error_msg)
