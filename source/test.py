# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import cv2
import sys

from parameters import Parameters
from network import Net

with open(sys.argv[1], 'r') as f:
	l = [line.split() for line in f]

zs = [[float(x) for x in line[:-1]] for line in l]
X_name = [line[-1] for line in l]

path_image = sys.argv[2]

p = Parameters()
p.NAME_OF_BEST_MODEL = '20181223-202436_gan.ckpt'

if p.NAME_OF_BEST_MODEL is None:
	raise Exception('Defina o modelo a ser carregado, no arquivo test.py')

# Inicializa a rede
n = Net(p)

for i in range(len(zs)):
	noise = np.array([zs[i]])
	img_name = X_name[i]
	n.image_generator(noise, path_image + '/' +img_name)
