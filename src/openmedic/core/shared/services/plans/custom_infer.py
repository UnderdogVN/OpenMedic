import logging
from typing import Union

import cv2
import numpy as np
import torch
import torchvision

from openmedic.core.shared.services.config import ConfigReader
from openmedic.core.shared.services.objects.model import (
    OpenMedicModel,
    OpenMedicModelBase,
)


class OpenMedicInferencerException(Exception):
    """Customizes exception"""

    def __init__(
        self, message: str = "An error occurred in OpenMedicInferencerException"
    ):
        self.message: str = message
        super().__init__(self.message)


class BaseInferencer:
    """Base class providing common inference utilities for all models."""

    def __init__(self, model: OpenMedicModelBase):
        self.model: OpenMedicModelBase = model

    def read_input(self, input_path: str, **kwargs) -> np.ndarray:
        logging.info(f"[inferencer][read_input]: Reading input from {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")
        return image

    def preprocess(
        self, image: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> torch.Tensor:
        target_shape = kwargs.get("target_shape", (512, 512))
        normalize = kwargs.get("normalize", True)
        mean = kwargs.get("mean", [0.485, 0.456, 0.406])
        std = kwargs.get("std", [0.229, 0.224, 0.225])

        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_shape)
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] != 3:
                image = image.permute(2, 0, 1)
            image = image.float()
            image = torchvision.transforms.functional.resize(image, target_shape)
        else:
            raise ValueError("Input data must be a numpy array or a torch tensor.")

        if normalize:
            image = torchvision.transforms.functional.normalize(
                image, mean=mean, std=std
            )

        # Add batch dimension if missing
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError("Processed image must be 3D or 4D after conversion.")

        return image

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        return output

    def postprocess(self, result):
        # TODO: Implement post-processing logic
        return result

    def run_inference(self, input_path: str, **kwargs):
        logging.info("[inferencer][run_inference]: Starting inference process...")
        image = self.read_input(input_path=input_path, **kwargs)
        processed_image = self.preprocess(image=image, **kwargs)
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
        model_checkpoint: str = model_info.get("model_checkpoint", "")
        model: OpenMedicModelBase = OpenMedicModel.get_model(model_name=model_name)(
            **model_params
        )
        if model_checkpoint:
            model.load_state_dict(torch.load(model_checkpoint))
        return cls(model)
