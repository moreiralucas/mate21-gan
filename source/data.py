# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import numpy as np
import cv2
import os
from parameters import Parameters

class Dataset:
    def __init__(self):
        self.scalares = np.ones(1)
        self.angulos = np.zeros(1)

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Load all images from a multiclass dataset (folder of folders). Each folder inside the main folder  #
    #         represents a different class and its name is used as class label. Train and test folders must have #
    #         the same directory structure, otherwise labels and their respective indexes will be misaligned.    #
    #         All images must have the same size, the same number of channels and 8 bits per channel.            #
    # Parameters:                                                                                                #
    #         path - path to the main folder                                                                     #
    #         height - number of image rows                                                                      #
    #         width - number of image columns                                                                    #
    #         num_channels - number of image channels                                                            #
    # Return values:                                                                                             #
    #         X - ndarray with all images                                                                        #
    #         y - ndarray with indexes of labels (y[i] is the label for X[i])                                    #
    #         l - list of existing labels (1st label in the list has index 0, 1nd has index 1, and so on)        #
    # ---------------------------------------------------------------------------------------------------------- #
    def load_multiclass_dataset(self, path, height=64, width=64, num_channels=1):
        classes = sorted(os.listdir(path))
        images = [sorted(os.listdir(path+'/'+id)) for id in classes]
        num_images = np.sum([len(l) for l in images])

        X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)
        y = np.empty([num_images], dtype=np.int64)
        print('x.shape: {}'.format(X.shape))
        print('y.shape: {}'.format(y.shape))

        k = 0
        for i in range(len(classes)):
            for j in range(len(images[i])):
                img = cv2.imread(path+'/'+classes[i]+'/'+images[i][j], cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
                assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[i][j]
                assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[i][j]
                X[k] = cv2.resize(img, (height, width)).reshape(height, width, num_channels)
                y[k] = i
                k += 1
        return [X, y], classes

    def load_images(self, path, height=224, width=224, num_channels=3):
        images = sorted(os.listdir(path))
        num_images = len(images)
        X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)
        y = np.empty([num_images], dtype=np.object)
        k = 0
        for j in range(len(images)):
                img = cv2.imread(path+'/'+images[j], cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
                assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[j]
                assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[j]
                X[k] = cv2.resize(img, (height, width)).reshape(height, width, num_channels)
                y[k] = str(images[j])
                k += 1
        return [X, y]
    
    def load_all_images(self, path_train, path_test, height=64, width=64, num_channels=1):
        data_train, _ = self.load_multiclass_dataset(path_train, height, width, num_channels)
        data_test = self.load_images(path_test, height, width, num_channels)
        return data_train[0] + data_test[0]

    def shuffle(self, X, y=None, seed=None):
        if y is not None:
            assert len(X) == len(y), "The 1st dimension size must be the same for both arrays!"
        if seed is not None:
            np.random.seed(seed)
        p = np.random.permutation(len(X))
        if y is not None:
            return X[p], y[p]
        else:
            return X[p]

    def split(self, X, y, rate):
        assert len(X) == len(y), "The 1st dimension size must be the same for both arrays!"
        idx = int(len(X)*float(rate))
        return [X[:idx], y[:idx]], [X[idx:], y[idx:]]

    def set_scales(self, scales):
        self.scalares = scales

    def set_angles(self, angles):
        self.angulos = angles

    def transform(self, img, angle=0, scale=1):
        p = Parameters()
        M = cv2.getRotationMatrix2D((p.IMAGE_WIDTH/2, p.IMAGE_HEIGHT/2), angle, scale)
        return cv2.warpAffine(img, M, (p.IMAGE_WIDTH, p.IMAGE_HEIGHT)).reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

    def augmentation(self, data, seed=42):
        scales = self.scalares
        angles = self.angulos
        p = Parameters()

        size_angles = len(angles)
        size_scales = len(scales)
        total_augmentation = int(size_scales * size_angles * 0.8)
        num_images = len(data[0]) * total_augmentation # + len(imgs)

        X_train = np.empty([num_images, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS], dtype=np.float32)
        y_train = np.empty([num_images], dtype=np.int64)
        k = 0
        np.random.seed(seed)
        for i in range(len(data[0])):
            for j in range(total_augmentation):
                scale = scales[np.random.randint(size_scales)]
                angle = angles[np.random.randint(size_angles)]
                # Adiciona imagens modificadas
                X_train[k] = self.transform(data[0][i], angle, scale) / 255.0
                # Adicona labels das imagens modificadas
                y_train[k] = data[1][i]
                k += 1 # Trocar esse k por j+ i
        
        print("len(X_train): {}".format(len(X_train)))
        print("len(y_train): {}".format(len(y_train)))
        x_out, y_out = self.shuffle(X_train, y_train, seed=42)
        return [x_out, y_out]
