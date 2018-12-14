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


class Net():
    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Load the training set, shuffle its images and then split them in training and validation subsets.  #
    #         After that, load the testing set.                                                                  #
    # ---------------------------------------------------------------------------------------------------------- #
    def __init__(self, input_train, p):
        self.train = input_train

        # ---------------------------------------------------------------------------------------------------------- #
        # Description:                                                                                               #
        #         Create a training graph that receives a batch of images and their respective labels and run a      #
        #         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
        #         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
        # ---------------------------------------------------------------------------------------------------------- #
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape = (None, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS))
            # self.y = tf.placeholder(tf.int64, shape = (None,))
            # self.y_one_hot = tf.one_hot(self.y, size_class_train)
            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            print(self.X.shape)
            with tf.variable_scope('encoder'): # Discriminator
                self.out = tf.layers.conv2d(self.X, 4, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

                self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='same')
                print(self.out.shape)

                self.out = tf.layers.conv2d(self.out, 16, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)
                self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='same')
                print(self.out.shape)

            with tf.variable_scope('decoder'): # Generator
                self.out = tf.layers.conv2d_transpose(self.out, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

                self.out = tf.layers.conv2d_transpose(self.out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

                self.out = tf.layers.conv2d_transpose(self.out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

            decoder_variables = [v for v in tf.global_variables() if v.name.startswith('decoder')]
            encoder_variables = [v for v in tf.global_variables() if v.name.startswith('encoder')]

            self.loss = tf.reduce_mean(tf.reduce_sum((self.out - self.X)**2))

            self.encoder_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=encoder_variables)
            self.decoder_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=decoder_variables)

            # self.result = tf.argmax(self.out, 1)
            # self.correct = tf.reduce_sum(tf.cast(tf.equal(self.result, self.y), tf.float32))

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Training loop.                                                                                     #
    # ---------------------------------------------------------------------------------------------------------- #

    def treino(self):
        p = Parameters()
        d = Dataset()
        with tf.Session(graph = self.graph) as session:
            # weight initialization
            session.run(tf.global_variables_initializer())

            menor_loss = 1e9
            best_acc = 0
            epoca = 0
            saver = tf.train.Saver()

            # full optimization
            for epoch in range(p.NUM_EPOCHS_FULL):
                print('\nEpoch: '+ str(epoch+1), end=' ')
                lr = (p.S_LEARNING_RATE_FULL*(p.NUM_EPOCHS_FULL-epoch-1)+p.F_LEARNING_RATE_FULL*epoch)/(p.NUM_EPOCHS_FULL-1)
                self.training_epoch(session, lr)
                # val_acc, val_loss = self.evaluation(session, self.val[0], self.val[1], name='Validation')
                test = d.load_N_images(p.TRAIN_FOLDER, seed=None)
                # print ('The model has successful saved')
                # cv2.imshow('input', test[5].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
                rec = session.run(self.out, feed_dict={self.X: test, self.is_training: False})
                # cv2.imshow('output', rec[0].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
                self.visualiza(rec, test)


            print ("Best_acc : " + str(best_acc) + ", loss: " + str(menor_loss) + ", epoca: " + str(epoca))

    def visualiza(self, rec, test):
        p = Parameters()
        d = Dataset()
        print("test.shape: ", end=' ')
        print(test.shape)

        cv2.imshow('in_0', test[0].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('in_1', test[1].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('in_2', test[2].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('in_3', test[3].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('in_4', test[4].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('in_5', test[5].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))

        cv2.imshow('out_0', rec[0].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('out_1', rec[1].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('out_2', rec[2].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('out_3', rec[3].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('out_4', rec[4].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        cv2.imshow('out_5', rec[5].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
        print(rec.shape)
        cv2.waitKey(1000)
        # ret1 = session.run([self.encoder_train_op, self.loss], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})
        # ret2 = session.run([self.decoder_train_op, self.loss], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Evaluate images in Xv with labels in yv.                                                           #
    # ---------------------------------------------------------------------------------------------------------- #
    def evaluation(self, session, Xv, yv, name='Evaluation'):
        p = Parameters()
        start = time.time()
        eval_loss = 0
        eval_acc = 0
        for j in range(0, len(Xv), p.BATCH_SIZE):
            ret = session.run([self.loss, self.correct], feed_dict = {self.X: Xv[j:j+p.BATCH_SIZE], self.y: yv[j:j+p.BATCH_SIZE], self.is_training: False})
            eval_loss += ret[0]*min(p.BATCH_SIZE, len(Xv)-j)
            eval_acc += ret[1]

        print(name+' Time:'+str(time.time()-start)+' ACC:'+str(eval_acc/len(Xv))+' Loss:'+str(eval_loss/len(Xv)))
        return eval_acc/len(Xv), eval_loss/len(Xv)

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Run one training epoch using images in X_train and labels in y_train.                              #
    # ---------------------------------------------------------------------------------------------------------- #
    def training_epoch(self, session, lr):
        batch_list = np.random.permutation(len(self.train))
        p = Parameters()
        start = time.time()
        train_loss1 = 0
        train_loss2 = 0
        k = 0
        print("batch:", end= ' ')
        for j in range(0, len(self.train), p.BATCH_SIZE):
            k += 1
            if j+p.BATCH_SIZE > len(self.train):
                break
            X_batch = self.train.take(batch_list[j:j+p.BATCH_SIZE], axis=0)

            ret1 = session.run([self.encoder_train_op, self.loss], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})
            ret2 = session.run([self.decoder_train_op, self.loss], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})

            train_loss1 += ret1[1]*p.BATCH_SIZE
            train_loss2 += ret2[1]*p.BATCH_SIZE
            print(k, end=' ')
        print("")

        pass_size = (len(self.train) - len(self.train) % p.BATCH_SIZE)
        print('LR:'+str(lr)+' Time:'+str(time.time()-start)+ ' Loss1:'+str(train_loss1/pass_size)+' Loss2:'+str(train_loss2/pass_size))
