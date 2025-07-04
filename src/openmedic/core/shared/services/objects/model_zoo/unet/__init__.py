# UNet model from https://github.com/milesial/Pytorch-UNet
import openmedic.core.shared.services.plans.registry as registry

from .model import UNet


def init():
    registry.ModelRegister.register(model_class=UNet)
