import os
import tempfile
import numpy as np
import cv2
import pydicom
import pytest
import json

from openmedic.core.shared.services.plans.custom_dataset import OpenMedicDataset

def create_minimal_coco_annotation():
    """Create a minimal valid COCO annotation file for testing."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    return coco_data

def test_read_png_and_return_rgb():
    # Use the actual PNG file from parent directory (not doubled path)
    test_dir = os.path.dirname(__file__)  # This is tests/test_read_dicom/
    img_dir = test_dir  # Look in the test directory itself
    
    # Create a temporary COCO annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(create_minimal_coco_annotation(), f)
        temp_coco_path = f.name
    
    try:
        ds = OpenMedicDataset(image_dir=img_dir, annotation_path=temp_coco_path, transform_ops=None)
        img = ds.read_image(img_dir, "sample_image.png")

        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3
        # dtype preserved from saved PNG (uint8)
        assert img.dtype == np.uint8
    finally:
        os.unlink(temp_coco_path)


def test_read_grayscale_dicom_converted_to_rgb():
    # Use the actual DICOM file from parent directory (not doubled path)
    test_dir = os.path.dirname(__file__)  # This is tests/test_read_dicom/
    img_dir = test_dir  # Look in the test directory itself
    
    # Create a temporary COCO annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(create_minimal_coco_annotation(), f)
        temp_coco_path = f.name
    
    try:
        ds = OpenMedicDataset(image_dir=img_dir, annotation_path=temp_coco_path, transform_ops=None)
        img = ds.read_image(img_dir, "sample_image.dcm")

        assert isinstance(img, np.ndarray)
        # converted to RGB => 3 channels
        assert img.ndim == 3 and img.shape[2] == 3
    finally:
        os.unlink(temp_coco_path)


def test_missing_file_raises_FileNotFoundError():
    test_dir = os.path.dirname(__file__)
    img_dir = test_dir
    
    # Create a temporary COCO annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(create_minimal_coco_annotation(), f)
        temp_coco_path = f.name
    
    try:
        ds = OpenMedicDataset(image_dir=img_dir, annotation_path=temp_coco_path, transform_ops=None)
        with pytest.raises(FileNotFoundError) as exc_info:
            ds.read_image(img_dir, "no_such_file.png")
        
        # Assert the error message contains expected information
        assert "no_such_file.png" in str(exc_info.value)
        assert "Image file not found" in str(exc_info.value)
    finally:
        os.unlink(temp_coco_path)


def test_unsupported_extension_raises_ValueError():
    test_dir = os.path.dirname(__file__)
    img_dir = test_dir
    bmp_path = os.path.join(img_dir, "test_file.bmp")
    
    # Create a temporary COCO annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(create_minimal_coco_annotation(), f)
        temp_coco_path = f.name
    
    # Create an empty file with unsupported extension
    with open(bmp_path, "wb") as f:
        f.write(b"")
    
    try:
        ds = OpenMedicDataset(image_dir=img_dir, annotation_path=temp_coco_path, transform_ops=None)
        with pytest.raises(ValueError) as exc_info:
            ds.read_image(img_dir, "test_file.bmp")
        
        assert "Unsupported image format" in str(exc_info.value)
        assert ".bmp" in str(exc_info.value)
    finally:
        # Clean up the test files
        if os.path.exists(bmp_path):
            os.remove(bmp_path)
        os.unlink(temp_coco_path)


if __name__ == "__main__":
	pytest.main([__file__])


