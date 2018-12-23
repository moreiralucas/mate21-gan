# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import cv2
import sys

from gan import Gan
from parameters import Parameters
from network import Net

with open(sys.argv[1], 'r') as f:
	l = [line.split() for line in f]

zs = [[float(x) for x in line[:-1]] for line in l]
X_name = [line[-1] for line in l]

path_image = sys.argv[2]


p = Parameters
p.param.NAME_OF_BEST_MODEL = 'Definir nome'

# Inicializa a rede
n = Net(p)
# Inicia treino


for i in range(len(zs)):
	noise = zs[i]
	img_name = X_name[i]
	n.image_generator(noise, img_name)