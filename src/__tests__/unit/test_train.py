"""
This module is use to test functions in the file train.py
"""
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# The sys.path.append helps us import all the module imports.
# The '..' depends on how many degrees the test file is away from main directory structure
from train import create_random_noise, data_loader, train_model
from models.encoder_decoder import ts_unet


def test_create_random_noise_shape():
    """Tests if the generated noise has the expected shape."""
    freq_bins = 100
    time_steps = 200
    noise = create_random_noise(freq_bins, time_steps)

    assert noise.shape == (freq_bins, time_steps)


def test_create_random_noise_values():
    """Tests if the generated noise values are within the expected range."""
    freq_bins = 10
    time_steps = 15
    noise = create_random_noise(freq_bins, time_steps)

    # Check if all values are between -1 and 1
    assert np.all(noise >= -1)
    assert np.all(noise <= 1)


def test_train_model():
    configs = {
        "timesteps": 100,
        "train_gen": 40,
        "optimizer": "Adam",
        "loss": "MeanSquaredError",
        "learning_rate": 0.01,
        "epochs": 0,
        "tsconv": False,
        "batch_size": 32,
    }
    train_data_path = None
    test_data_path = None
    model = train_model(train_data_path, test_data_path, configs)
    # Check if the model is returned
    assert model is not None
