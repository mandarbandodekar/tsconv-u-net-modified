"""
This module is used to test TSConv Class
"""

import pytest
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models.tsconv import TSConv


@pytest.fixture
def tsconv_layer():
    """
    Instantiate TSConv Class
    """
    return TSConv(filters=32, kernel_size=(3, 1))


def test_tsconv_initialization(tsconv_layer):
    """
    Test initialisation of TSConv class
    :param
    tsconv_layer: pytest fixture
    """
    assert tsconv_layer.filters == 32
    assert tsconv_layer.kernel_size == (3, 1)


def test_tsconv_build(tsconv_layer):
    """
    This functions check TSConv build layer
    :param
    tsconv_layer: pytest fixture
    """
    input_shape = (1, 10, 10, 1)
    tsconv_layer.build(input_shape)
    assert isinstance(tsconv_layer.conv, tf.keras.layers.Conv2D)


def test_tsconv_call(tsconv_layer):
    """
    This function checks for Modified TSConv time shifted layers
    :param
    :param
    tsconv_layer: pytest fixture
    """
    input_shape = (1, 10, 10, 4)
    inputs = tf.random.normal(input_shape)
    outputs = tsconv_layer(inputs)
    assert outputs.shape == (input_shape[0], 10, 10, 32)
