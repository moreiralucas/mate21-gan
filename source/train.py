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
    # Inicializa e configura parâmetros
    p = Parameters()
    d = Dataset()
    # d.set_scales(np.linspace(0.7, 1.3, 5))
    # d.set_angles(np.linspace(-10, 10, 3))

    # Carrega as imagens do treino e do test com suas respectivas labels
    train = d.load_all_images(p.TRAIN_FOLDER, p.TEST_FOLDER)
    train = train[:2000]
    train = train / 255.0
    
    print("size of train: {}".format(len(train)))
    
    # Embaralhas as imagens
    train = d.shuffle(train, seed=42)
    
    print(train.shape)
        
    # Inicializa a rede
    n = Net(train, p)
    # Inicia treino
    n.treino()

if __name__ == "__main__":
    main()