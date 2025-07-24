import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import torch

from openmedic.core.shared.services.config import ConfigReader
from openmedic.core.shared.services.objects.model import (
    OpenMedicModel,
    OpenMedicModelBase,
)
from openmedic.core.shared.services.objects.transform import (
    OpenMedicTransform,
    OpenMedicTransformOpBase,
)
from openmedic.core.shared.services.utils import get_current_time


class OpenMedicInferencerException(Exception):
    """Customizes exception"""

    def __init__(
        self,
        message: str = "An error occurred in OpenMedicInferencerException",
    ):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicInferencer:
    """Inferencer class for OpenMedic models."""

    def __init__(self, model: OpenMedicModelBase):
        self.model: OpenMedicModelBase = model
        self.inference_info: dict = ConfigReader.get_field(name="pipeline")
        self.transform_ops = self._plan_transform()

        if self.inference_info.get("mask_threshold") is None:
            self.mask_threshold = 0.3
            logging.info(
                f"Mask threshold is not set in config, using default: {self.mask_threshold}",
            )

    @classmethod
    def initialize_with_config(cls):
        """Initializes the inferencer using model configuration.
        Note: ConfigReader must be initialized beforehand.
        """
        model_info: dict = ConfigReader.get_field(name="model")
        model_name: str = model_info["name"]
        model_params: dict = model_info["params"]
        model_checkpoint: str = model_info.get("model_checkpoint", "")
        model: OpenMedicModelBase = OpenMedicModel.get_model(model_name=model_name)(
            **model_params,
        )

        if model_checkpoint:
            model.load_state_dict(torch.load(model_checkpoint))

        return cls(model)

    def _plan_transform(self) -> list:
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

    def read_input(self, input_path: str) -> np.ndarray:
        logging.info(f"[inferencer][read_input]: Reading input from {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")
        return image

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image_copy = image.copy()
        if isinstance(image, np.ndarray):
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError("Input image must be a numpy array or a torch tensor.")

        for op in self.transform_ops:
            if isinstance(op, OpenMedicTransformOpBase):
                image = op.execute_inference(image_copy)
            else:
                logging.warning(
                    "[inferencer][preprocess]: No transform operations applied.",
                )
        # Add batch dimension if missing
        if image_copy.ndim == 3:
            image_copy = torch.from_numpy(image_copy).permute(
                2,
                0,
                1,
            )
            image_copy = image_copy.float()
            image_copy = image_copy.unsqueeze(0)
        elif image_copy.ndim != 4:
            raise ValueError("Processed image must be 3D or 4D after conversion.")

        if self.inference_info["is_gpu"]:
            image_copy = image_copy.to(torch.device("cuda"))
        return image_copy

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        if self.inference_info["is_gpu"]:
            self.model = self.model.to(torch.device("cuda"))
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        return output

    def postprocess(self, output) -> np.ndarray:
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [192, 192, 192],
            [128, 128, 128],
            [128, 0, 0],
            [128, 128, 0],
            [0, 128, 0],
            [128, 0, 128],
            [0, 128, 128],
            [0, 0, 128],
        ]

        _, channels, height, width = output.shape
        channel_colors = np.array(colors[:channels], dtype=np.float32)

        pred_image = np.ones((height, width, 3), dtype=np.float32) * 255
        for y in range(height):
            for x in range(width):
                selected_colors = channel_colors[
                    output[0, :, y, x] > self.mask_threshold
                ]

                if len(selected_colors) > 0:
                    pred_image[y, x, :] = np.mean(selected_colors, axis=0)

        return pred_image.astype(np.uint8)

    def save_image(self, image: np.ndarray):
        image_name = os.path.basename(self.inference_info["input_path"])
        if (
            self.inference_info["output_dir"] == ""
            or self.inference_info["output_path"] is None
        ):
            from openmedic.core.shared.services.plans.management import OpenMedicOSEnv

            output_dir = os.path.join(
                OpenMedicOSEnv.home,
                "inference_" + get_current_time().strftime("%Y%m%d.%H%M%S"),
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"pred_{image_name}",
            )
        else:
            output_path = os.path.join(
                self.inference_info["output_dir"],
                f"pred_{image_name}",
            )
        cv2.imwrite(
            output_path,
            image,
        )
        logging.info(
            f"[inferencer][save_image]: Image saved to {output_path}",
        )

    def run_inference(self, **kwargs):
        logging.info("[inferencer][run_inference]: Starting inference process...")
        image = self.read_input(input_path=self.inference_info["input_path"])
        pred_image = self.preprocess(image=image)
        output = self.inference(pred_image)
        output = self.postprocess(output)
        self.save_image(image=output)
        logging.info("[inferencer][run_inference]: Inference process completed.")
        return output
