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

def discriminator(X, reuse_variables=None, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse_variables): # discriminator -> encoder
        out = tf.layers.dense(X, 256, activation=tf.nn.relu)
        out = tf.layers.dense(out, 1, activation=None)
    return out

class Net():
    def __init__(self, input_train, p):
        self.train = input_train
        self.graph = tf.Graph()
        self.param = p
        self.shape_out = (None, 32, 32, 1)

        with self.graph.as_default():
            self.ph_gen = tf.placeholder(tf.float32, shape = (None, 64))
            self.ph_dis = tf.placeholder(tf.float32, shape = (None, self.param.IMAGE_HEIGHT, self.param.IMAGE_WIDTH, self.param.NUM_CHANNELS))

            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            
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

    def _get_noise(self, batch_size, noise_dim=64):
        mu, sigma = 0, 1.0 # mean and standard deviation
        z = np.random.normal(mu, sigma, size=[batch_size, noise_dim])
        return z

    def treino(self):
        p = Parameters()
        d = Dataset()
        with tf.Session(graph = self.graph) as session:
            
           # Tensorboard area
            loss_gen = tf.placeholder(tf.float32, shape=())
            loss_dis = tf.placeholder(tf.float32, shape=())
            img_pl = tf.placeholder(tf.float32, shape=self.shape_out)

            score_summary_op = tf.summary.merge([
                tf.summary.scalar('Generator_loss', loss_gen),
                tf.summary.scalar('Discriminator_loss_real', loss_dis),
                tf.summary.image('Generated_images', img_pl, 9)
            ])
            logdir = self.param.TENSORBOARD_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(logdir, session.graph)

            # weight initialization
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # full optimization
            for epoch in range(self.param.NUM_EPOCHS_FULL):
                print('Epoch: '+ str(epoch+1), end=' ')
                
                lr = (self.param.S_LEARNING_RATE_FULL*(self.param.NUM_EPOCHS_FULL-epoch-1)+self.param.F_LEARNING_RATE_FULL*epoch)/(self.param.NUM_EPOCHS_FULL-1)
                loss1, loss2, img_vis = self._training_epoch(session, lr)

                if epoch % 20 == 0:
                    # print("Salvou as imagens!")
                    # self.visualiza_and_save(img_vis, epoch)
                    scores_summary = session.run(
                        score_summary_op,
                        feed_dict={
                            loss_gen: loss2,
                            loss_dis: loss1,
                            img_pl: img_vis,
                            self.is_training: False
                        })
                    writer.add_summary(scores_summary, global_step=epoch)
                    writer.flush()

            path_model = self.param.LOG_DIR_MODEL  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_gan.ckpt'
            saver.save(session, path_model)
            print("The model has saved in: " + path_model)

    def _training_epoch(self, session, lr):
        batch_list = np.random.permutation(len(self.train))
        p = Parameters()
        start = time.time()
        train_loss1 = 0
        train_loss2 = 0
        img = None
        NEW_BATCH = self.param.BATCH_SIZE//2

        for j in range(0, len(self.train), NEW_BATCH):
            if j+NEW_BATCH > len(self.train):
                break

            x_batch = self.train.take(batch_list[j:j+NEW_BATCH], axis=0)
            x_noise = self._get_noise(NEW_BATCH)
            ret1 = session.run([self.discriminator_train_op, self.loss_dis], feed_dict = {self.ph_dis: x_batch, self.ph_gen: x_noise, self.learning_rate: lr})
            
            x_noise = self._get_noise(self.param.BATCH_SIZE)
            ret2 = session.run([self.generator_train_op, self.loss_gen, self.out_ruido], feed_dict = {self.ph_gen: x_noise, self.learning_rate: lr})

            img = ret2[2]
            train_loss1 += ret1[1]*self.param.BATCH_SIZE
            train_loss2 += ret2[1]*self.param.BATCH_SIZE

        pass_size = (len(self.train) - len(self.train) % self.param.BATCH_SIZE)
        print('LR:'+str(lr  )+' Time:'+str(time.time()-start)+ ' Loss_dis:'+str(train_loss1/pass_size)+' Loss_gen:'+str(train_loss2/pass_size))
        return train_loss1, train_loss2, img

    def visualiza_and_save(self, imgs, ep):
        N = len(imgs)
        cont = 0
        for img in imgs:
            if cont % 64 == 0:
                img *= 255.0
                path = "output/img" + str(ep) + "_" + str(cont) + ".png"
                img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)
                cv2.imwrite(path, img)
            cont += 1

    def image_generator(self, input_noise, height=64, width=64, num_channels=1):
        p = Parameters()
        path_model = self.param.LOG_DIR_MODEL + self.param.NAME_OF_BEST_MODEL
        path_img_output = 'output/'

        with tf.Session(graph = self.graph) as session:
            saver = tf.train.Saver(max_to_keep=0)
            saver.restore(session, path_model)
            #num_images = len(input_noise)

            #predicao = np.empty([num_images, height, width, num_channels], dtype=np.uint8)
            for noise in input_noise:
                image_name = noise[64] # Recupera o nome da imagem que é o último elemento da lista
                input_noise = input_noise[:64] # ou é input_noise[:64 - 1]?

                feed_dict = {self.ph_gen: noise}
                ret = session.run([self.out_ruido], feed_dict)
                print("ret.shape" + str(ret.shape))

                self._save_image(ret, path_img_output + str(image_name))

    def _save_image(self, img, path='output/img.png'):
        print(path)
        print("img.shape: " + str(img.shape))
        # height, width, channels = img.shape
        # img = img.reshape(height, width)
        # img = cv2.resize(img, (self.param.IMAGE_HEIGHT, self.param.IMAGE_WIDTH))
        # cv2.imwrite(path, img)
