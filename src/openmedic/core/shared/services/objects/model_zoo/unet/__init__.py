# UNet model from https://github.com/milesial/Pytorch-UNet
from .model import UNet
import openmedic.core.shared.services.plans.registry as registry


def init():
    registry.ModelRegister.register(model_class=UNet)