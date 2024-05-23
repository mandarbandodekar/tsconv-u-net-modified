"""
This module is use to test functions in the file parameter_validations.py
"""
import numpy as np
import sys
import pytest
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# The sys.path.append helps us import all the module imports.
# The '..' depends on how many degrees the test file is away from main directory structure

from parameter_validations import data_shape_validator, assertions


@pytest.mark.parametrize("config", [
    # Valid configurations
    {"optimizer": "Adam", "loss": "mse", "epochs": 10,
     "learning_rate": 0.01, "batch_size": 32},
    {"optimizer": "SGD", "loss": "binary_crossentropy", "epochs": 20,
     "learning_rate": 0.001, "batch_size": 64},
])
def test_valid_assertions(config):
    """
    checks whether valid assertions work as expected
    :params
    config: [dict]: Required config.json parameters
    """
    assertions(config)


@pytest.mark.parametrize("config, error_message", [
    # Invalid optimizer
    ({"optimizer": "not_an_optimizer", "loss": "mse", "epochs": 10,
      "learning_rate": 0.01, "batch_size": 32},
     "Invalid optimizer"),
    # Invalid loss
    ({"optimizer": "Adam", "loss": "not_a_loss", "epochs": 10,
      "learning_rate": 0.01, "batch_size": 32},
     "Invalid loss"),
    # Invalid epochs
    ({"optimizer": "Adam", "loss": "mse", "epochs": 0,
      "learning_rate": 0.01, "batch_size": 32},
     "Invalid number of epochs defined"),
    # Invalid learning rate
    ({"optimizer": "Adam", "loss": "mse", "epochs": 10,
      "learning_rate": 0.0, "batch_size": 32},
     "Invalid learning rate"),
    # Invalid batch size
    ({"optimizer": "Adam", "loss": "mse", "epochs": 10,
      "learning_rate": 0.01, "batch_size": 0},
     "Invalid batch size"),
])
def test_invalid_assertions(config, error_message):
    """
    Raises error
    :param
    config [dict]: dictionary of
    """
    with pytest.raises(Exception) as except_error:
        assertions(config)
    assert error_message in str(except_error.value)


@pytest.mark.parametrize("train_array, test_array, expected_train, expected_test", [
    # Valid cases: One dimension of either train or test data is 256
    ((100, 256), (50, 256), (256, 100), (256, 50)), 
    ((256, 100), (256, 50), (256, 100), (256, 50))
])
def test_data_shape_validator(train_array, test_array, expected_train, expected_test):
    """
    This test check whether outputs of the array are transposed as per required
    and of the inputs is of shape 256 or not
    :param
    train_array: [Tuple]
    test_array: [Tuple]
    expected_train: [Tuple]
    expected_test: [Tuple]
    """
    train_result, test_result = data_shape_validator(np.random.rand(train_array[0],train_array[1]) * 2 - 1,
                                                     np.random.rand(test_array[0],test_array[1]) * 2 - 1)
    assert train_result.shape == expected_train
    assert test_result.shape == expected_test


@pytest.mark.parametrize("train_array, test_array", [
    ((50, 100), (16, 50)),
    ((256, 100), (256, 50)),
])
def test_data_shape_error(train_array, test_array):
    """
    This test checks whether pytest raises value error for incorrect 
    array shapes
    :param
    train_array: [Tuple]
    test_array: [Tuple]
    """
    with pytest.raises(ValueError):
        data_shape_validator(np.array(train_array), np.array(test_array))