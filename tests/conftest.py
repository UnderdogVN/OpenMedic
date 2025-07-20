import pytest


@pytest.fixture
def five_fundus_train() -> str:
    return "examples/manifest_files/five_binary_train.yml"


@pytest.fixture
def five_fundus_eval() -> str:
    return "examples/manifest_files/five_binary_eval.yml"


@pytest.fixture
def manifest_anatomy() -> dict:
    anatomy = {
        "correct_manifest_train": {
            "data": {
                "image_dir": "openmedic/image_dir",
                "coco_annotation_path": "openmedic/coco_annotation_path"
            },
            "model": {
                "name": "openmendic_model",
                "params": {
                    "n_channels": 1,
                    "n_classes": 1
                }
            },
            "pipeline": {
                "batch_size": 1,
                "n_epochs": 1,
                "train_ratio": 1
            },
            "optimization": {
                "name": "openmendic_optimizer",
                "params": {
                    "lr": 0.0001,
                    "random_params": "opemendic_optimizer_random_params"
                }
            },
            "loss_function": {
                "name": "openmedic_loss_function",
                "type": "custom",
                "params": {
                    "reduction": "mean",
                    "weight": [1, 2]
                },
                "extra_param": "test"
            },
            "metric": {
                "name": "opemendic_metric",
                "params": {}
            },
            # Optional fields
            "transform": {
                "openmedic_transfrom": {}
            },
            "monitor": {
                "opemendic_monitor": {}
            }
        },
        "correct_manifest_eval": {
            "data": {
                "image_dir": "openmedic/image_dir",
                "coco_annotation_path": "openmedic/coco_annotation_path"
            },
            "model": {
                "name": "openmendic_model",
                "params": {
                    "n_channels": 1,
                    "n_classes": 1
                },
                "model_checkpoint": "openmedic/model_checkpoint.pth"
            },
            "pipeline": {
                "batch_size": 1,
                "n_epochs": 1,
                "train_ratio": 1
            },
            "loss_function": {
                "name": "openmedic_loss_function",
                "type": "custom",
                "params": {
                    "reduction": "mean",
                    "weight": [1, 2]
                },
                "extra_param": "test"
            },
            "metric": {
                "name": "opemendic_metric",
                "params": {}
            },
            # Optional fields
            "transform": {
                "openmedic_transfrom": {}
            },
            "monitor": {
                "opemendic_monitor": {}
            }
        }
    }

    return anatomy