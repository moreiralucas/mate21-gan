import tensorflow as tf
import numpy as np

class Parameters:
    def __init__(self):
        # Image
        self.IMAGE_HEIGHT = 32  # height of the image / rows
        self.IMAGE_WIDTH = 32   # width of the image / cols
        self.NUM_CHANNELS = 1   # number of channels of the image
        # Database
        self.TRAIN_FOLDER = '../data_part1/train' # folder with training images
        self.TEST_FOLDER = '../data_part1/test'   # folder with testing images
        self.SPLIT_RATE = 0.80        # split rate for training and validation sets
        self.COUNT_NOISE = 64
        # Training loop
        self.LOG_DIR_MODEL = './model/'
        self.TENSORBOARD_DIR = 'tensorboard/'
<<<<<<< HEAD
        self.NUM_EPOCHS_FULL = 4000
=======
        self.NUM_EPOCHS_FULL = 1000
>>>>>>> dc0e314f4f02cd998f5fae4d38c7dbb95df03dac
        self.LEARNING_RATE_DISC = 1e-4
        self.LEARNING_RATE_GEN = 1e-4
        self.F_LEARNING_RATE_FULL = 0.001
        self.BATCH_SIZE = 64
        self.TOLERANCE = 10
        # Model
        self.NAME_OF_BEST_MODEL = None
        # Others parameters
