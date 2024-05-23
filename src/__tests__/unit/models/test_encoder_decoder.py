"""
This module defines different tests for functions of encoder-decoder module
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models.encoder_decoder import encoder_block, decoder_block, ts_unet


@pytest.mark.parametrize("choice", [True, False])
def test_encoder_block(choice):
    """
    Encoder block tests
    :param
    choice: [boolean] True or False
    """
    # Define input tensor
    # Arrange
    input_shape = (1, 10, 10, 4)
    inputs = tf.random.normal(input_shape)
    conv1 = Conv2D(4, (3, 1), activation="relu", padding="same")(inputs)

    # Define number of filters
    filters = 32

    # Call encoder_block function
    # Act
    output = encoder_block(conv1, filters, choice)

    # Verify output shape
    # Assert
    assert output.shape == (1, 5, 10, filters)


@pytest.mark.parametrize("choice", [True, False])
def test_decoder_block(choice):
    """
    Decoder block tests
    :param
    choice: [boolean] True or False
    """
    # Define input tensor
    input_shape = (1, 10, 10, 4)
    inputs = tf.random.normal(input_shape)
    other_shape = (1, 20, 10, 4)

    other_layer = tf.random.normal(other_shape)

    # Define number of filters
    filters = 32

    # Call encoder_block function
    output = decoder_block(inputs, other_layer, filters, choice)

    # Verify output shape
    # The assertions is as per Unet architecture defined
    assert output.shape == (1, 20, 10, filters)


@pytest.mark.parametrize("choice", [True, False])
def test_ts_unet(choice):
    """
    Unet architecture tests
    :param
    choice: [boolean] True or False
    """

    # Arrange
    input_shape = (256, 10, 1)
    # Act
    model = ts_unet(input_shape, choice)

    # assert
    assert model is not None
    # Check Final shape is similar to input shape
    assert (None, 256, 10, 1) == model.layers[-1].output_shape
