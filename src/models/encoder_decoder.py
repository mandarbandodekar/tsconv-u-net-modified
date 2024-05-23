"""
This module defines the Modified U-net encoder decoder architecture.

Functions
- ```encoder_block```: Executes encoder block of Given Unet architecture
- ```decoder_block```: Executes decoder block of Given Unet architecture
- ```ts_unet```: Keras layers as per the defined Modified Unet architecure
"""

import tensorflow as tf
from .tsconv import TSConv
from tensorflow.keras.layers import (
    Input, Conv2D
)


def encoder_block(inputs, filters, choice):
    """
    Encoder block of a modified Unet.

    :param
    inputs [tf.Tensor]: Input tensor to the encoder block.
    filters [int]: Number of filters for the convolutional layers.
    choice [boolean]: Choice of convolution network or Tsconv.

    :return
    conv2  (tf.Tensor): Output tensor after passing through the encoder block.
    """

    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(inputs)
    if choice:
        conv1 = TSConv(filters)(pool)

        conv2 = TSConv(filters)(conv1)
    else:
        conv1 = Conv2D(filters, (3, 3), activation="relu",
                       padding="same")(pool)
        conv2 = Conv2D(filters, (3, 3), activation="relu",
                       padding="same")(conv1)

    return conv2


def decoder_block(inputs, other_layer, filters, choice):
    """
    Decoder block of the modified Unet.

    :param
    inputs [tf.Tensor]: Input tensor to the encoder block.
    other_layer [tf.Tensor]: The layer with which the input
                            needs to be concatenated
    filters [int]: Number of filters for the convolutional layers.
    choice [boolean]: Choice of convolution network or Tsconv.

    :return
    conv2 [tf.Tensor]: Output tensor after passing through the decoder block.

    """
    upsample = tf.keras.layers.UpSampling2D(size=(2, 1))(inputs)
    concat = tf.keras.layers.Concatenate()([upsample, other_layer])

    if choice:
        conv1 = TSConv(filters)(concat)
        conv2 = TSConv(filters)(conv1)
    else:
        conv1 = Conv2D(filters, (3, 3), activation="relu",
                       padding="same")(concat)
        conv2 = Conv2D(filters, (3, 3), activation="relu",
                       padding="same")(conv1)

    return conv2


def ts_unet(input_shape, choice):
    """
    The Unet Architecture.
    :param
    input_shape [Tuple] : Input shape of the tensor
    choice [boolean]: Choice of convolution network or Tsconv.
    :return
    model : tf.keras Trained Keras U-net model
    """
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(4, (3, 1), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(4, (3, 1), activation="relu", padding="same")(conv1)

    encoder_layer2 = encoder_block(conv1, 4, choice)
    encoder_layer3 = encoder_block(encoder_layer2, 4, choice)
    encoder_layer4 = encoder_block(encoder_layer3, 8, choice)
    encoder_layer5 = encoder_block(encoder_layer4, 8, choice)
    encoder_layer6 = encoder_block(encoder_layer5, 16, choice)

    # Decoder
    decoder_layer1 = decoder_block(encoder_layer6, encoder_layer5, 8, choice)
    decoder_layer2 = decoder_block(decoder_layer1, encoder_layer4, 8, choice)
    decoder_layer3 = decoder_block(decoder_layer2, encoder_layer3, 4, choice)
    decoder_layer4 = decoder_block(decoder_layer3, encoder_layer2, 4, choice)
    decoder_layer5 = decoder_block(decoder_layer4, conv1, 4, choice)

    # Output
    output = Conv2D(1, (1, 1), activation="sigmoid")(decoder_layer5)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
