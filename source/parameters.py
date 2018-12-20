import tensorflow as tf
import numpy as np
# import random
# import time
# import sys
# import os

# from data import Dataset

class Parameters:
    def __init__(self):
        # Image
        self.IMAGE_HEIGHT = 64  # height of the image / rows
        self.IMAGE_WIDTH = 64   # width of the image / cols
        self.NUM_CHANNELS = 1   # number of channels of the image
        # Database
        self.TRAIN_FOLDER = 'data_part1/train' # folder with training images
        self.TEST_FOLDER = 'data_part1/test'   # folder with testing images
        self.SPLIT_RATE = 0.80        # split rate for training and validation sets
        # Training loop
        self.LOG_DIR_MODEL = 'model/'
        self.TENSORBOARD_DIR = 'tensorboard/'
        self.NUM_EPOCHS_FULL = 50000
        self.S_LEARNING_RATE_FULL = 0.001
        self.F_LEARNING_RATE_FULL = 0.001
        self.BATCH_SIZE = 256
        self.TOLERANCE = 10
        # Others parameters
        # here ...
