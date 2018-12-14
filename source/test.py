# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import cv2
import sys

from gan import Gan

with open(sys.argv[1], 'r') as f:
	l = [line.split() for line in f]
zs = [[float(x) for x in line[:-1]] for line in l]
X_name = [line[-1] for line in l]
X_test = np.array(zs).reshape(-1, 8, 8, 1)
print(X_name)
print(X_test.shape, X_test.dtype)