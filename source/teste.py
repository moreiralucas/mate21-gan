import tensorflow as tf
import numpy as np
import random
import math
import cv2
import random

from data import Dataset
from parameters import Parameters
d = Dataset()
d.set_scales(np.linspace(0.7, 1.3, 5))
d.set_angles(np.linspace(-10, 10, 3))

p = Parameters()
# Carrega as imagens do treino com suas respectivas labels
# train, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE).reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

imagens = []
for i in range(10):
   imagens.append(img)

print(imagens[0].shape)
imagens = np.array(imagens)

radian = []
for i in range(10):
    degree_angle = np.random.randint(0, 360)
    radian.append(degree_angle * math.pi / 180)

radian = np.array(radian)
tf_img = tf.contrib.image.rotate(imagens, radian)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rotated_img = sess.run(tf_img)
    # print(rotated_img)
    for k in range(len(rotated_img)):
        # cv2.imshow("Original", img)
        # cv2.imshow("Rotacionada", rotated_img[k])
        # cv2.waitKey(0)
        pat = "output/o_" + str(k) + ".png"
        cv2.imwrite(pat, rotated_img[k])
    # print(rotated_img[0].shape)
    
    img_view = rotated_img
    cv2.destroyAllWindows()