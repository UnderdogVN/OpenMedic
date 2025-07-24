from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import openmedic.core.shared.services.plans.registry as registry
from openmedic.core.shared.services.objects.transform import (
    OpenMedicTransformOpBase,
    OpenMedicTransformOpError,
)


@dataclass
class Resize(OpenMedicTransformOpBase):
    __LIMIT_SIZE: int = 100
    # Ground truth interpolation must to be fixed `INTER_NEAREST` to
    # keeps the generic pixel value of the labels
    __GT_INTERPOLATION: int = cv2.INTER_NEAREST

    def __init__(self, target_w: int, target_h: int, interpolation: str):
        self.target_w: int = target_w
        self.target_h: int = target_h
        self.cv2_interpolation: any = getattr(cv2, interpolation)

    @classmethod
    def initialize(cls, *args, **kwargs):
        limit_size: int = Resize.__LIMIT_SIZE
        target_w: Optional[int] = kwargs.get("target_w", None)
        target_h: Optional[int] = kwargs.get("target_h", None)
        interpolation: Optional[str] = kwargs.get("interpolation", None)

        if not target_w or not target_h:
            raise OpenMedicTransformOpError(
                "`target_w` or `target_h` does not exist in cofig file.",
            )
        elif target_w < limit_size or target_h < limit_size:
            raise OpenMedicTransformOpError(
                f"`target_w` or `target_h` needs to greater or equal than {limit_size}.",
            )

        if not interpolation:
            # Set default interpolation.
            interpolation = "INTER_LINEAR"

        return cls(target_w, target_h, interpolation)

    def execute(self, image: np.ndarray, gt: np.ndarray) -> List[np.ndarray]:

        image_copy: np.ndarray = image.copy()
        gt_copy: np.ndarray = gt.copy()

        image_copy = cv2.resize(
            src=image_copy,
            dsize=(self.target_w, self.target_h),
            interpolation=self.cv2_interpolation,
        )

        gt_copy = cv2.resize(
            src=gt_copy,
            dsize=(self.target_w, self.target_h),
            interpolation=self.__GT_INTERPOLATION,  # with ground truth image we need apply `INTER_NEAREST`
        )

        return image_copy, gt_copy

    def execute_inference(self, image: np.ndarray) -> np.ndarray:
        """Resize the image for inference."""
        image_copy: np.ndarray = image.copy()
        image_copy = cv2.resize(
            src=image_copy,
            dsize=(self.target_w, self.target_h),
            interpolation=self.cv2_interpolation,
        )
        return image_copy


def init():
    registry.TransformRegister.register(transform_class=Resize)
