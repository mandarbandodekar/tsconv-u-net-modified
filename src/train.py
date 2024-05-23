"""
This module executes training code for Conv2d and TSConv networks

Functions:
- ```train_model```: Executes the Conv2D or TSConv training model
- ``` data_loader``: Loads the data if defined else create random data
                    for execution
- ```create_random_noise```: Create Random data if no data specified
"""

import tensorflow as tf

# ==== Python Standard imports ====
import numpy as np
import argparse
import json
import os
from datetime import datetime

# ==== Module imports ====
from models.encoder_decoder import ts_unet
from parameter_validations import assertions, data_shape_validator


def train_model(train_data_path, segmented_data_path, configs):
    """
    Train the custom Unet model.
    This functions compiles and Trains the custom Unet model

    :param
    train_data_path [str]:  Path to the directory containing training data.
    filters: Number of filters for the convolutional layers.
    configs: A list of Neural network Hyperparameters as well as
            other parameters required for the training model

    :return None
    """
    train_images, segmented_images = data_loader(
        train_data_path, segmented_data_path, configs
    )
    input_shape = (256, train_images.shape[2], 1)

    # Since we have fixed frequency bins
    model = ts_unet(input_shape, configs["tsconv"])
    model.summary()

    # Parse optimizer and loss from string to corresponding TensorFlow objects
    optimizer = getattr(tf.keras.optimizers, configs["optimizer"])()
    loss = getattr(tf.keras.losses, configs["loss"])()
    optimizer.learning_rate.assign(configs['learning_rate'])
    print(optimizer.learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    # Train the model
    model.fit(
        train_images,
        epochs=configs["epochs"],
        batch_size=configs["batch_size"],
        y=segmented_images,
    )

    return model


def data_loader(train_data_path, segmented_data_path, configs):
    """
    The data loader is used for mock data
    :param
    train_data_path [str]: Data directory where train/test data is located
    configs
    :return
    spectro_train_images[np.ndarray]:  Numpy array  Train data
    spectro_segmented_images [np.ndarray]: Numpy array Test data
    """

    if train_data_path is None:
        spectro_train_images = []
        spectro_segmented_images = []
        # "The Frequency bins needs to be 256"
        for i in range(configs["train_gen"]):
            spectro_train_images.append(
                create_random_noise(256, configs["timesteps"])
            )
            spectro_segmented_images.append(
                create_random_noise(256, configs["timesteps"])
            )
    else:
        # Since no data loader is available, the data needs to be as .npy
        try:
            spectro_train_images = np.load(train_data_path)
            spectro_segmented_images = np.load(segmented_data_path)
        except ValueError as e:
            raise f"The data input needs to be numpy array, {e}"

        spectro_train_images, spectro_segmented_images = data_shape_validator(
            spectro_train_images, spectro_segmented_images
        )
    spectro_train_images = np.expand_dims(spectro_train_images, axis=-1)
    spectro_segmented_images = np.expand_dims(spectro_segmented_images,
                                                  axis=-1)

    return spectro_train_images, spectro_segmented_images


def create_random_noise(frequency, timesteps):
    """
    Generates a random noise spectrogram with specified frequency bins and
    time steps.
    :param
    frequency [int]: It is strictly defined as 256
    timesteps [int]: Number of timesteps aka audio length of spectrogram image
    """
    # Generate random data between -1 and 1
    noise = np.random.rand(frequency, timesteps) * 2 - 1

    return noise


def main():
    """
    This function is main executable function and is
    responsible for the following:
    - Reading argparse paramters
    - Validating config.json hyperparameters
    - Executing training function module

    :param None
    :return None
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a TensorFlow model")

    # Add command line arguments
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="Path to the directory containing numpy training data",
    )
    parser.add_argument(
        "--segmented_data_path",
        type=str,
        default=None,
        help="Path to numpy segmented training output data directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--config_json", type=str, default="config.json",
        help="Parameters Path"
    )
    parser.add_argument(
        "--use_TSConv",
        action="store_true",
        help="Use TSConvolution for unet segmentation",
    )
    parser.add_argument(
        "--model_output_dir", type=str, default=None,
        help="Directory to save models"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Print parsed arguments
    print(f"- Data directory: {args.train_data_path}")
    print(f"- Batch size: {args.batch_size}")

    # Data Loads
    config_path = args.config_json
    with open(config_path) as f:
        configs = json.load(f)

    if args.use_TSConv:
        configs["tsconv"] = True
        print("arg true")
    else:
        configs["tsconv"] = False
        print("arg false")

    configs["batch_size"] = args.batch_size
    print(configs)

    assertions(configs)
    # this function checks for all parameters whether they are defined as per requirements

    # Train the model
    model = train_model(args.train_data_path,
                        args.segmented_data_path, configs)
    now = datetime.now()
    model_name = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    model_output_dir = args.model_output_dir
    if model_output_dir is not None:
        if not os.path.isdir(model_output_dir):
            # If path does not exists set model directory to None
            model_output_dir = None
    if model_output_dir is not None:
        output_model_path = os.path.join(model_output_dir, model_name + ".keras")
    else:
        output_model_path = os.path.join("output-logs", model_name + ".keras")

    model.save(output_model_path)


if __name__ == "__main__":
    main()
