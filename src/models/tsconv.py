"""
This module consist of modified TSConv layer class that reduces the number of
parameters by 1/3 compared to traditional convolutional networks
"""
import tensorflow as tf


class TSConv(tf.keras.layers.Layer):
    """
    This class implements a custom TSConv layer,
    :param
    filters: [int] Number of filters for the convolutional layer.
    kernel_size: [tuple] (optional) Size of the convolutional kernel.
                Defaults to (3, 1).

    Methods:
    * `__init__()`: Initializes the layer with the specified filters, choice,
                    and kernel size.
    * `build(self, input_shape)`: Builds the layer by creating a
                                `tf.keras.layers.Conv2D` instance.
    * `call(self, inputs)`: Performs the forward pass of the layer. It split
                            the input into static and dynamic components,
                            performs shifting on the dynamic components,
                            concatenates them, and applies
                            a 2D convolution.
    """
    def __init__(self, filters, kernel_size=(3, 1)):
        super(TSConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(self.filters,
                                           self.kernel_size,
                                           activation='relu',
                                           padding='same')

    def call(self, inputs):
        static, dynamic = tf.split(inputs, num_or_size_splits=2, axis=-1)
        dynamic_neg,dynamic_pos = tf.split(dynamic, num_or_size_splits=2,
                                           axis=-1)
        dynamic_shift_pos = tf.roll(dynamic_pos, shift=1, axis=1)
        dynamic_shift_neg = tf.roll(dynamic_neg, shift=-1, axis=1)
        x = tf.concat([static, dynamic_shift_pos, dynamic_shift_neg], axis=-1)
        return self.conv(x)
