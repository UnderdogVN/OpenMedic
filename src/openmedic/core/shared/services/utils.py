import datetime
import importlib
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from typing import List

import numpy as np
import yaml
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


class ModuleInterface:
    """The plugin interface"""

    @staticmethod
    def init() -> None:
        """Registers the Module"""


def camel_to_snake(val: str) -> str:
    """Converts CamelCase or camelCase to snake_case.
    Example: "CamelCase" -> "camel_case"

    Input:
    ------
        val: str - The CamelCase value.

    Output:
    -------
        str - The snake_case value.
    """
    s1: str = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", val)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """Converts snake_case to CamelCase.
    Example: "camel_case" -> "CamelCase"

    Input:
    ------
        val: str - The snake_case value.

    Output:
    -------
        str - The CamelCase value.
    """
    components: list = name.split("_")
    return "".join(x.capitalize() for x in components)


def load_coco_file(annotation_path: str) -> COCO:
    """Loads COCO file.

    Input:
    ------
        annotation_path: str - The annotation file path.

    Output:
    -------
        COCO - The constructor of Microsoft COCO.
    """
    # Suppress stdout by COCO
    with io.StringIO() as buf, redirect_stdout(buf):
        coco: COCO = COCO(annotation_file=annotation_path)
    return coco


def convert_to_gt(ann_ids: List[dict], img_h: int, img_w) -> np.ndarray:
    """Converts the COCO annotation to ground truth image.

    Input:
    ------
        ann_ids: List[dict] - The COCO annotation.
        img_h: int - The image's height.
        img_w: int - The image's width.

    Output:
    ------
        gt_canvas: np.ndarray - The grouth truth image as array.


    Usage:
    ------
    ```
        from pycocotools.coco import COCO
        img_id: int
        coco: COCO
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        gt: np.ndarray = utils.convert_to_gt(
            ann_ids=ann_ids,
            img_h=img_info["height"],
            img_w=img_info["width"]
        )
    ```
    """
    gt_canvas: np.ndarray = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann_id in ann_ids:
        seg: any = ann_id["segmentation"]
        category_id: int = ann_id["category_id"]
        mask: np.ndarray
        if isinstance(seg, dict):  # RLE
            if isinstance(seg["counts"], str):
                seg["counts"] = seg["counts"].encode("utf-8")
            mask = maskUtils.decode(seg)
        else:  # Polygon
            rles = maskUtils.frPyObjects(seg, img_h, img_w)
            mask = maskUtils.decode(rles)
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=2)  # merge multiple polygons into one

        # Make the mask pixel value according to catgory_id
        mask[mask == 1] = category_id
        gt_canvas = np.maximum(gt_canvas, mask.astype(np.uint8))

    return gt_canvas


def import_module(module_name: str) -> any:
    """Imports Python module.

    Input:
    ------
        module_name: str - The module name.

    Output:
    ------
        any - The Python module.
    """
    if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
        # https://bobbyhadz.com/blog/python-importerror-cannot-import-name-mapping-from-collections
        import collections.abc

        collections.Mapping = collections.abc.Mapping
        collections.MutableMapping = collections.abc.MutableMapping
        return importlib.import_module(module_name)


class BreakLoop(Exception):
    """Raises exception to break the loop."""

    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


def get_current_time() -> datetime.datetime:
    return datetime.datetime.now()


def save_as_yml(data: dict, file_path: str, if_exist: str = "append"):
    mode: str = "w"
    if os.path.isfile(file_path) and if_exist == "append":
        mode = "a"

    if not file_path.endswith(".yml") and not file_path.endswith(".yaml"):
        raise Exception(f"Only support with file format `.yml` or `.yaml`.")

    with open(file_path, mode) as file:
        if mode == "a":
            file.write("\n---\n")  # YAML document separator
        yaml.dump(data, file)


def save_as_json(data: dict, file_path: str, if_exist: str = ""):
    if not file_path.endswith(".json"):
        raise Exception(f"[save_as_json]: Only support with file format `.json`.")

    if os.path.isfile(file_path) and not if_exist:
        raise Exception(
            f"[save_as_json]: The file `file_path` ({file_path}) is exist! Need to set `if_exist` = 'overwrite'.",
        )

    with open(file_path, "w") as file:
        json.dump(
            data,
            file,
            indent=4,
        )  # indent=4 makes the JSON file pretty and readable (indented by 4 spaces).
