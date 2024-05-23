"""
This Python file checks all the parameters are valid and within requirements
Functions:
- ```assertions```: Validates all hyperparameters and parameter configs
- ```data_shape_validator```: Checks whether data shape has the required
                            data size as per defined Unet architecture
"""

import tensorflow as tf


def assertions(configs):
    """
    This file checks all the hyperparameters as well as arg parse parameters
    are defined as per requirements
    :param
    configs: dictionary of config.json
    return: None
    """
    assert hasattr(
        tf.keras.optimizers, configs["optimizer"]
    ), f"""Invalid optimizer: {configs['optimizer']},
        Please check https://keras.io/api/optimizers/ for proper optimizers"""

    # Assert that loss is a valid loss class
    assert hasattr(
        tf.keras.losses, configs["loss"]
    ), f"""Invalid loss: {configs['loss']}, Please check
        https://keras.io/api/losses/regression_losses/
        for proper regression loss function definitions"""

    # Assert that epochs is a positive integer
    assert (
        configs["epochs"] > 0
    ), f"Invalid number of epochs defined: {configs['epochs']}. The Epochs need to be positive integer"

    # Assert that learning_rate is a positive float
    assert (
        isinstance(configs["learning_rate"], float) and configs["learning_rate"] > 0
    ), f"Invalid learning rate: {configs['learning_rate']}"

    # Assert that batch_size is a positive integer
    assert (
        isinstance(configs["batch_size"], int) and configs["batch_size"] > 0
    ), f"Invalid batch size: {configs['batch_size']}"


def data_shape_validator(train_np_array, test_np_array):
    """
    This functions check image data shape. As the defined Unet architecture
    has a fixed frequency of 256.
    One of the dimensions of the train and test images needs to be 256.
    :param
    train_np_array [np.ndArray]: Train data from data directory
    test_np_array  [np.ndArray]: Test data from data directory
    :return
    train_np_array [np.ndArray]: Validated Train Numpy array shape
    test_np_array  [np.ndArray]: Validated Test Numpy array shape
    """

    if (256 not in train_np_array.shape) and (256 not in test_np_array.shape):
        raise ValueError(
            """Neither training nor test images have a dimension of (256, ...).
            The Unet architecture has frequency of 256"""
        )

    if train_np_array.shape[0] != 256:
        train_np_array = train_np_array.transpose()

    if test_np_array.shape[0] != 256:
        test_np_array = test_np_array.transpose()

    return train_np_array, test_np_array
