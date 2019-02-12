# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os

from data import Dataset
from parameters import Parameters
from network import Net

def main():
    # os.nice(20)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Inicializa e configura parâmetros
    p = Parameters()
    d = Dataset()


    # Carrega as imagens do treino e do test com suas respectivas labels
    train = d.load_all_images(p.TRAIN_FOLDER, p.TEST_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH)
<<<<<<< HEAD
    #train = train[:5000]
=======
>>>>>>> dc0e314f4f02cd998f5fae4d38c7dbb95df03dac
    train = train / 255.0
    
    print("size of train: {}".format(len(train)))
    
    # Embaralhas as imagens
    train = d.shuffle(train, seed=42)
    
    print(train.shape)
#    p.NUM_EPOCHS_FULL = 10
    # Inicializa a rede
    n = Net(p)
    # Inicia treino
    n.treino(train)


if __name__ == "__main__":
    main()
