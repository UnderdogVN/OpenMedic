import logging
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import openmedic.core.shared.services.utils as utils
from openmedic.core.shared.services.config import ConfigReader
from openmedic.core.shared.services.objects.transform import (
    OpenMedicTransform,
    OpenMedicTransformOpBase,
)


class OpenMedicDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_path: str,
        transform_ops: Optional[List[OpenMedicTransformOpBase]],
    ):
        """
        Input:
        ------
            image_dir: str - Directory of images.
            annotation_path: str - Annotation file (COCO format).
            transform_ops (Optional): List[OpenMedicTransformOpBase] - List of openmedic transfom operators.
        """
        self.image_dir: str = image_dir
        self.annotation_path: str = annotation_path
        self.coco: COCO = utils.load_coco_file(annotation_path=annotation_path)
        self.img_ids: list = list(self.coco.imgs.keys())
        self.transform_ops: Optional[List[OpenMedicTransformOpBase]] = transform_ops
        self.is_tensor: bool = False

    def get_categories(self) -> List[dict]:
        """Gets list of category."""
        return self.coco.loadCats(self.coco.getCatIds())

    def set_return_tensor(self, flag: bool):
        """Sets to returns torch.Tensor or np.ndarray in method `__getitem__`

        Input:
        ------
            flag: bool - If True then returns torch.Tensor otherwise returns np.ndarray.
        """
        self.is_tensor = flag

    def get_num_classes(self) -> int:
        """Gets number of classes in dataset.
        Note: `num_classes = all category + negative class (background)`.
        """
        return (
            len(self.get_categories()) + 1
        )  # num_classes = all category + negative class (background)

    @classmethod
    def initialize_with_config(cls):
        data_info: dict = ConfigReader.get_field(name="data")
        return cls(
            data_info["image_dir"],
            data_info["coco_annotation_path"],
            cls._plan_transform(),
        )

    @staticmethod
    def _plan_transform() -> list:
        transform_plan_ops: Optional[dict] = ConfigReader.get_field(name="transform")
        if not transform_plan_ops:
            return []

        op_names: list = []
        op_name: str
        params: any
        transform_ops: List[OpenMedicTransformOpBase] = []
        for op_name, params in transform_plan_ops.items():
            if isinstance(params, dict):
                transform_ops.append(
                    OpenMedicTransform.get_op(op_name=op_name).initialize(**params),
                )
            elif isinstance(params, list):
                transform_ops.append(
                    OpenMedicTransform.get_op(op_name=op_name).initialize(*params),
                )
            else:
                transform_ops.append(
                    OpenMedicTransform.get_op(op_name=op_name).initialize([params]),
                )

            op_names.append(op_name)

        logging.info(
            f"[DatasetManager][_plan_transform]: Applied {', '.join(op_names)} operators.",
        )

        return transform_ops

    def _apply_transform(self, image: np.ndarray, gt: np.ndarray) -> List[np.ndarray]:
        image_transform = image.copy()
        gt_transform = gt.copy()
        for transform_op in self.transform_ops:
            image_transform, gt_transform = transform_op.execute(
                image=image_transform,
                gt=gt_transform,
            )

        return image_transform, gt_transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> List[Union[np.ndarray, torch.Tensor]]:
        img_id: int = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))

        gt: np.ndarray = utils.convert_to_gt(
            ann_ids=ann_ids,
            img_h=img_info["height"],
            img_w=img_info["width"],
        )
        image: np.ndarray = cv2.imread(
            os.path.join(self.image_dir, img_info["file_name"]),
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform_ops:
            image, gt = self._apply_transform(image=image, gt=gt)

        if self.is_tensor:
            image_tensor: torch.Tensor = torch.from_numpy(image)
            gt_tensor: torch.Tensor = torch.from_numpy(gt)
            return image_tensor, gt_tensor

        return image, gt
