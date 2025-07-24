import copy

import pytest

import src.openmedic.core.shared.services.config as config


def test_read_manifest(five_fundus_train: str, five_fundus_eval: str):
    # Read manifest train
    config.ConfigReader.initialize(config_path=five_fundus_train, mode="train")
    model_config: dict = config.ConfigReader.get_field("model")
    expected_output: set = {"name", "params"}
    assert (
        set(model_config.keys()) == expected_output
    ), "Incorrect model's fields in train"

    # Read manifest evaluation
    config.ConfigReader.initialize(config_path=five_fundus_eval, mode="eval")
    model_config: dict = config.ConfigReader.get_field("model")
    expected_output: set = {"name", "params", "model_checkpoint"}
    assert (
        set(model_config.keys()) == expected_output
    ), "Incorrect model's fields in evaluation"


def test_render_train(manifest_anatomy):
    ### Correct cases ###
    correct_manifest_train: dict = manifest_anatomy["correct_manifest_train"]
    config.ConfigReader.init_manifest(config=correct_manifest_train, mode="train")

    data_config: dict = config.ConfigReader.get_field("data")
    expected_output: set = {"image_dir", "coco_annotation_path"}
    assert (
        set(data_config.keys()) == expected_output
    ), "Incorrect data's fields in train"

    ### Incorrect cases ###
    incorrect_manifest_train: dict

    # Case 1: miss requisite field
    incorrect_manifest_train = copy.deepcopy(correct_manifest_train)
    del incorrect_manifest_train["data"]
    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_train, mode="train")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException("Config validation failed: 1 validation error for ManifestTrainer\\ndata\\n  Field required [type=missin...mendic_monitor\': {}}}, input_type=dict]\\n    For further information visit https://errors.pydantic.dev/2.11/v/missing") tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"

    # Case 2: incorrect train ratio
    incorrect_manifest_train = copy.deepcopy(correct_manifest_train)

    incorrect_manifest_train["pipeline"]["train_ratio"] = 0
    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_train, mode="train")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException('Config validation failed: 1 validation error for ManifestTrainer\\npipeline.train_ratio\\n  Input shoul...an, input_value=0, input_type=int]\\n    For further information visit https://errors.pydantic.dev/2.11/v/greater_than') tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"

    incorrect_manifest_train["pipeline"]["train_ratio"] = 1.1
    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_train, mode="train")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException('Config validation failed: 1 validation error for ManifestTrainer\\npipeline.train_ratio\\n  Input shoul...ut_value=1.1, input_type=float]\\n    For further information visit https://errors.pydantic.dev/2.11/v/less_than_equal') tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"

    # Case 3: miss requisite parameter
    incorrect_manifest_train = copy.deepcopy(correct_manifest_train)

    del incorrect_manifest_train["data"]["image_dir"]

    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_train, mode="train")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException("Config validation failed: 1 validation error for ManifestTrainer\\ndata.image_dir\\n  Field required [t...oco_annotation_path\'}, input_type=dict]\\n    For further information visit https://errors.pydantic.dev/2.11/v/missing") tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"


def test_render_eval(manifest_anatomy):
    ### Correct cases ###
    correct_manifest_eval: dict = manifest_anatomy["correct_manifest_eval"]
    config.ConfigReader.init_manifest(config=correct_manifest_eval, mode="eval")

    data_config: dict = config.ConfigReader.get_field("data")
    expected_output: set = {"image_dir", "coco_annotation_path"}
    assert (
        set(data_config.keys()) == expected_output
    ), "Incorrect data's fields in train"

    ### Incorrect cases ###
    incorrect_manifest_eval: dict

    # Case 1: miss requisite field
    incorrect_manifest_eval = copy.deepcopy(correct_manifest_eval)
    del incorrect_manifest_eval["data"]
    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_eval, mode="eval")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException("Config validation failed: 1 validation error for ManifestEvaluator\\ndata\\n  Field required [type=miss...mendic_monitor\': {}}}, input_type=dict]\\n    For further information visit https://errors.pydantic.dev/2.11/v/missing") tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"

    # Case 2: miss requisite parameter
    incorrect_manifest_eval = copy.deepcopy(correct_manifest_eval)
    del incorrect_manifest_eval["pipeline"]["batch_size"]
    with pytest.raises(Exception) as error_msg:
        config.ConfigReader.init_manifest(config=incorrect_manifest_eval, mode="eval")

    excepted_error_msg: str = (
        """<ExceptionInfo ConfigException("Config validation failed: 1 validation error for ManifestEvaluator\\npipeline.batch_size\\n  Field requ... 1, \'train_ratio\': 1}, input_type=dict]\\n    For further information visit https://errors.pydantic.dev/2.11/v/missing") tblen=2>"""
    )
    assert excepted_error_msg == str(error_msg), "Incorrect error message"


def test_render_infer(manifest_anatomy):
    # TODO: Need to implement here
    ### Correct cases ###

    ### Incorrect cases ###
    pass
