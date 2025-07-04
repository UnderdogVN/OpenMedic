from dataclasses import dataclass
from typing import List

import numpy as np

import openmedic.core.shared.services.plans.registry as registry
from openmedic.core.shared.services.objects.transform import OpenMedicTransformOpBase


@dataclass
class Flip(OpenMedicTransformOpBase):
    def __init__(self):
        pass

    @classmethod
    def initialize(cls, *args, **kwargs):
        return cls()

    def execute(self, image: np.ndarray, gt: np.ndarray) -> List[np.ndarray]:
        # TODO: Need to implement logic
        return image, gt


def init():
    registry.TransformRegister.register(transform_class=Flip)
