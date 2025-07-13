import logging
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


class OpenMedicInferencerException(Exception):
    """Customizes exception"""

    def __init__(
        self,
        message: str = "An error occurred in OpenMedicInferencerException",
    ):
        self.message: str = message
        super().__init__(self.message)


class BaseInferencer:
    """Base class providing common inference utilities for all models."""

    def __init__(self, model: OpenMedicModelBase):
        self.model: OpenMedicModelBase = model
        self.inference_info: dict = ConfigReader.get_field(name="inference")
        self.transform_ops = self._plan_transform()

    def _plan_transform(self) -> list:
        transform_plan_ops: Optional[dict] = self.inference_info.get("transform", None)
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

        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError("Input image must be a numpy array or a torch tensor.")

        for op in self.transform_ops:
            if isinstance(op, OpenMedicTransformOpBase):
                image = op.execute_inference(image=image)
            else:
                logging.warning(
                    "[inferencer][preprocess]: No transform operations applied.",
                )
        # Add batch dimension if missing
        if image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert HWC to CHW
            image = image.float()  # Convert to float tensor
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError("Processed image must be 3D or 4D after conversion.")

        if self.inference_info["device"] == "cuda":
            image = image.to(torch.device("cuda"))
        return image

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        if self.inference_info["device"] == "cuda":
            self.model = self.model.to(torch.device("cuda"))
        with torch.no_grad():
            output = self.model(image)
        return output

    def postprocess(self, result):
        # TODO: Implement post-processing logic
        return result

    def run_inference(self, input_path: str, **kwargs):
        logging.info("[inferencer][run_inference]: Starting inference process...")
        image = self.read_input(input_path=input_path)
        processed_image = self.preprocess(image=image)
        result = self.inference(processed_image)
        result = self.postprocess(result)
        logging.info("[inferencer][run_inference]: Inference process completed.")
        return result


class OpenMedicInferencer(BaseInferencer):
    """Inferencer using configuration initialization."""

    @classmethod
    def initialize_with_config(cls):
        """Initializes the inferencer using model configuration.
        Note: ConfigReader must be initialized beforehand.
        """
        model_info: dict = ConfigReader.get_field(name="model")
        model_name: str = model_info["name"]
        model_params: dict = model_info["params"]
        model_checkpoint: str = model_info.get("model_path", "")
        model: OpenMedicModelBase = OpenMedicModel.get_model(model_name=model_name)(
            **model_params,
        )

        if model_checkpoint:
            model.load_state_dict(torch.load(model_checkpoint))

        model.eval()
        return cls(model)
