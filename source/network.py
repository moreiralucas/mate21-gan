import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os
import cv2

from data import Dataset
from parameters import Parameters

def generator(X):
    with tf.variable_scope('generator'): # generator
        out_img = tf.layers.dense(X, 128, activation=tf.nn.relu)
        out_img = tf.layers.dense(out_img, 256, activation=tf.nn.sigmoid)
        out_img = tf.reshape(out_img, (-1, 16, 16, 1))
    return out_img

# self.X = tf.placeholder(tf.float32, shape = (None, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS))
def discriminator(X, reuse_variables=None):
    with tf.variable_scope('discriminator', reuse=reuse_variables): # discriminator -> encoder
        out = tf.layers.dense(X, 256, activation=tf.nn.relu)
        out = tf.layers.dense(out, 1, activation=None)
    return out

p = Parameters()

class Net():
    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Load the training set, shuffle its images and then split them in training and validation subsets.  #
    #         After that, load the testing set.                                                                  #
    # ---------------------------------------------------------------------------------------------------------- #
    def __init__(self, input_train, p):
        self.train = input_train
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.ph_gen = tf.placeholder(tf.float32, shape = (None, 64))
            self.ph_dis = tf.placeholder(tf.float32, shape = (None, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS))

            self.learning_rate = tf.placeholder(tf.float32)
            # self.is_training = tf.placeholder(tf.bool)
            
            # meio batch discriminator(real) + meio batch pro generator
            self.out_real = discriminator(self.ph_dis)
            self.out_ruido = generator(self.ph_gen)
            self.out_fake = discriminator(self.out_ruido, reuse_variables=True)

            discriminator_variables = [v for v in tf.global_variables() if v.name.startswith('discriminator')]
            generator_variables = [v for v in tf.global_variables() if v.name.startswith('generator')]

            self.loss_dis_r = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.out_real, labels = tf.ones_like(self.out_real))))
            self.loss_gen = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.out_ruido, labels = tf.ones_like(self.out_ruido))))
            self.loss_dis_f = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.out_fake, labels = tf.zeros_like(self.out_fake))))

            self.loss_dis = self.loss_dis_f + self.loss_dis_r

            self.discriminator_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_dis, var_list=discriminator_variables)
            self.generator_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_gen, var_list=generator_variables)

    def _get_noise(self, batch_size):
        mu, sigma = 0, 1.0 # mean and standard deviation
        noise_dim = 64
        z = np.random.normal(mu, sigma, size=[batch_size, noise_dim])
        return z

    def treino(self):
        p = Parameters()
        d = Dataset()
        with tf.Session(graph = self.graph) as session:
            """
            # Tensorboard area
            tf.summary.scalar('Generator_loss', self.loss_gen)
            tf.summary.scalar('Discriminator_loss_real', self.loss_dis_r)
            tf.summary.scalar('Discriminator_loss_fake', self.loss_dis_f)

            # imgs_to_tb = generator(self.ph_gen, re=tf.AUTO_REUSE)
            tf.summary.image('Generated_images', [None, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS], 6)
            self.merged = tf.summary.merge_all()
            logdir = p.TENSORBOARD_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            self.writer = tf.summary.FileWriter(logdir, session.graph)
            """
            # weight initialization
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # full optimization
            for epoch in range(p.NUM_EPOCHS_FULL):
                print('\nEpoch: '+ str(epoch+1), end=' ')
                
                lr = (p.S_LEARNING_RATE_FULL*(p.NUM_EPOCHS_FULL-epoch-1)+p.F_LEARNING_RATE_FULL*epoch)/(p.NUM_EPOCHS_FULL-1)
                img_vis = self._training_epoch(session, lr)
            
            path_model = p.LOG_DIR_MODEL  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_gan.ckpt'
            saver.save(session, path_model)
            print("The model has saved in: " + p.LOG_DIR_MODEL)

    def _training_epoch(self, session, lr):
        batch_list = np.random.permutation(len(self.train))
        p = Parameters()
        start = time.time()
        train_loss1 = 0
        train_loss2 = 0
        img = None
        print("batch:", end= ' ')
        NEW_BATCH = p.BATCH_SIZE//2

        for j in range(0, len(self.train), NEW_BATCH):
            if j+NEW_BATCH > len(self.train):
                break

            x_batch = self.train.take(batch_list[j:j+NEW_BATCH], axis=0)
            x_noise = self._get_noise(NEW_BATCH)
            ret1 = session.run([self.discriminator_train_op, self.loss_dis], feed_dict = {self.ph_dis: x_batch, self.ph_gen: x_noise, self.learning_rate: lr})
            
            x_noise = self._get_noise(p.BATCH_SIZE)
            ret2 = session.run([self.generator_train_op, self.loss_gen, self.out_ruido], feed_dict = {self.ph_gen: x_noise, self.learning_rate: lr})

            img = ret2[2]
            train_loss1 += ret1[1]*p.BATCH_SIZE
            train_loss2 += ret2[1]*p.BATCH_SIZE

        pass_size = (len(self.train) - len(self.train) % p.BATCH_SIZE)
        print('LR:'+str(lr  )+' Time:'+str(time.time()-start)+ ' Loss_dis:'+str(train_loss1/pass_size)+' Loss_gen:'+str(train_loss2/pass_size))
        return img

    def visualiza_and_save(self, imgs, ep):
        N = len(imgs)
        cont = 0
        for img in imgs:
            cont += 1
            if cont % 2 == 0:
                continue 
            if cont == 10:
                break
            path = "output/img" + str(ep) + "_" + str(cont) + ".png"
            img = cv2.resize(img, (0,0), fx=2.0, fy=2.0) 
            cv2.imwrite(path, img)