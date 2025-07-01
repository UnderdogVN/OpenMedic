from typing import Dict
import yaml
import argparse
from abc import ABC, abstractmethod
import logging


class ConfigException(Exception):
    """Customs exception"""
    def __init__(self, message: str="An error occurred in ConfigReader"):
        self.message: str = message
        super().__init__(self.message)


class ConfigReader:
    @classmethod
    def _init(cls, sections: Dict[str, dict]):
        for section,  content in sections.items():
            setattr(cls, section, content)

    @classmethod
    def initialize(cls, config_path: str):
        """Sets attributes for `ConfigReader` by reading configure file.

        Input:
        ------
            config_path: str - The configure path.
        """
        if not config_path.endswith(".yml") and \
            not config_path.endswith(".yaml"):
            raise ConfigException("Only support `yaml` or `yml` file.")

        with open(config_path, 'r') as stream:
            cls._init(sections=yaml.safe_load(stream))


    @classmethod
    def _check_required_field(cls, name: str, attr_fields: list):
        """Check the permanent attributes of field.
        If the config file includes incorrect attributes, then raises an error.
        Input:
        ------
            name: str - The field name.
            attr_fields: List - The list of attribute values of field.
        """
        REQUIRED_FIELD_MAPPING: dict = {
            "data": ["image_dir", "coco_annotation_path"],
            "model": ["name", "params"],
            "loss_function": ["name", "params", "type"],
            "optimization": ["name", "params"],
            "metric": ["name", "params"],
            "pipeline": ["batch_size", "n_epochs", "train_ratio"]
        }
        required_fields: list = REQUIRED_FIELD_MAPPING.get(name, [])

        is_subset: bool = set(required_fields).issubset(set(attr_fields))
        if not is_subset:
            error_msg = f"The required fields in `{name}`: {required_fields}"
            raise ConfigException(message=error_msg)

    @classmethod
    def get_field(cls, name: str) -> any:
        """Get the attributes of field in the config file.
        Input:
        ------
            name: str - The field name.

        Output:
        ------
            attr: any - The attribute of field.
        """
        VERIFIABLE_FIELD: list = [
            "transform",
            "data",
            "model",
            "optimization",
            "loss_function",
            "pipeline",
            "metric",
            "monitor"
        ]
        error_msg: str = ''
        try:
            attr: any =  getattr(cls, name)
            if name in VERIFIABLE_FIELD:
                if not isinstance(attr, dict):
                    error_msg = f"The field `{name}` need to be parsed as a dictionary."
                    raise ConfigException(message=error_msg)

                attr_fields: list = list(attr.keys())
                cls._check_required_field(name=name, attr_fields=attr_fields)
            return attr

        except AttributeError:
            if name in ["transform", "monitor"]:
                # Return None if config file does not include `transform` or `monitor` field
                logging.warning(f"[ConfigReader][get_field]: The field `{name}` is not set in the config file.")
                return None
            error_msg = f"The field `{name}` does not exist in the config file."
            raise ConfigException(message=error_msg)