import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os

from data import Dataset
from parameters import Parameters


class Net():
    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Load the training set, shuffle its images and then split them in training and validation subsets.  #
    #         After that, load the testing set.                                                                  #
    # ---------------------------------------------------------------------------------------------------------- #
    def __init__(self, input_train, input_val, p, size_class_train=10):
        self.train = input_train
        self.val = input_val

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
            with tf.variable_scope('encoder'):
                # self.X = tf.layers.dropout(self.X, 0.2, training=self.is_training) # Dropout
                self.out = tf.layers.conv2d(self.X, 4, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
                print(self.out.shape)
                
                self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='same')
                print(self.out.shape)

                self.out = tf.layers.conv2d(self.out, 16, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)
            
                self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='same')
                print(self.out.shape)
            
            with tf.variable_scope('decoder'):
                self.out = tf.layers.conv2d_transpose(self.out, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

                self.out = tf.layers.conv2d_transpose(self.out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
                print(self.out.shape)

            decoder_variables = [v for v in tf.global_variables() if v.name.startswith('decoder')]
            encoder_variables = [v for v in tf.global_variables() if v.name.startswith('encoder')]
            
            print(decoder_variables)
            print(encoder_variables)
                        
            # self.out = tf.layers.max_pooling2d(self.out, (3, 3), (2, 2), padding='valid')

            # self.out = tf.layers.dropout(self.out, 0.3, training=self.is_training) # Dropout
            # print(self.out.shape)

            # self.out = tf.reshape(self.out, [-1, self.out.shape[1]*self.out.shape[2]*self.out.shape[3]])

            # self.out = tf.layers.dense(self.out, size_class_train, activation=tf.nn.sigmoid)

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
        with tf.Session(graph = self.graph) as session:
            # weight initialization
            session.run(tf.global_variables_initializer())

            menor_loss = 1e9
            best_acc = 0
            epoca = 0
            saver = tf.train.Saver()

            # full optimization
            for epoch in range(p.NUM_EPOCHS_FULL):
                print('Epoch: '+ str(epoch+1), end=' ')
                lr = (p.S_LEARNING_RATE_FULL*(p.NUM_EPOCHS_FULL-epoch-1)+p.F_LEARNING_RATE_FULL*epoch)/(p.NUM_EPOCHS_FULL-1)
                self.training_epoch(session, lr)
                # val_acc, val_loss = self.evaluation(session, self.val[0], self.val[1], name='Validation')
                
                # Otimizar o early stopping
                if val_acc > best_acc:
                    menor_loss = val_loss
                    best_acc = val_acc
                    epoca = epoch
                    saver.save(session, os.path.join(p.LOG_DIR, 'model.ckpt'))
                    print ('The model has successful saved')
                cv2.imshow('input', self.train.reshape(p.IMAGE_HEIGHT, p;IMAGE_WIDTH))
                rec = session.run(self.out, feed_dict={X: self.train[0:0+1], is_training: False})
                cv2.imshow('output', rec[0].reshape(p.IMAGE_HEIGHT, p.IMAGE_WIDTH))
                cv2.waitKey(0)
                print ('\n-********************************************************-')

            print ("Best_acc : " + str(best_acc) + ", loss: " + str(menor_loss) + ", epoca: " + str(epoca)) 
    
    # def prediction(self, test, classes_train):
    #     print ('-********************************************************-')
    #     print ('Start prediction ...')
    #     #p = Parameters()
    #     with tf.Session(graph = self.graph) as session:
    #         outputs = None
    #         time_now = datetime.datetime.now()
    #         path_txt = str(time_now.day) + '_' + str(time_now.hour) + 'h'  + str(time_now.minute) + 'm.txt'
    #         with open(path_txt, 'w') as f:
    #             for j in range(len(test[0])):
    #                 feed_dict={self.X: np.reshape(test[0][j], (1, ) + test[0][j].shape), self.is_training: False}
    #                 saida = session.run(self.out, feed_dict)
    #                 outputs = np.array(saida[0])
    #                 resp = str(test[1][j]) +' ' + str(np.argmax(outputs)) + '\n'
    #                 f.write(resp)
    #             f.close()
    
    # def prediction2(self, test, classes_train):
    #     p = Parameters()
    #     tf.reset_default_graph()
    #     saver = tf.train.Saver()
    #     with tf.Session() as session:
    #         saver.restore(session, os.path.join(p.LOG_DIR, 'model.ckpt'))  
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

        for j in range(0, len(self.train), p.BATCH_SIZE):
            if j+p.BATCH_SIZE > len(self.train):
                break
            X_batch = self.train.take(batch_list[j:j+p.BATCH_SIZE], axis=0)

            ret1 = session.run([self.encoder_train_op, self.loss, self.correct], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})
            ret2 = session.run([self.decoder_train_op, self.loss, self.correct], feed_dict = {self.X: X_batch, self.learning_rate: lr, self.is_training: True})
            
            train_loss1 += ret1[1]*p.BATCH_SIZE
            train_loss2 += ret2[1]*p.BATCH_SIZE

        pass_size = (len(self.train) - len(self.train) % p.BATCH_SIZE)
        print('LR:'+str(lr)+' Time:'+str(time.time()-start)+ ' Loss1:'+str(train_loss1/pass_size)' Loss2:'+str(train_loss2/pass_size))






# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os
import cv2

from data import load_multiclass_dataset, shuffle, split

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Parameters.                                                                                        #
# ---------------------------------------------------------------------------------------------------------- #
TRAIN_FOLDER = './train' # folder with training images
TEST_FOLDER = './test'   # folder with testing images
SPLIT_RATE = 0.90        # split rate for training and validation sets

IMAGE_HEIGHT = 64  # height of the image
IMAGE_WIDTH = 64   # width of the image
NUM_CHANNELS = 1   # number of channels of the image

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load the training set, shuffle its images and then split them in training and validation subsets.  #
#         After that, load the testing set.                                                                  #
# ---------------------------------------------------------------------------------------------------------- #
X_train, y_train, classes_train = load_multiclass_dataset(TRAIN_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
X_train = X_train/255.#.reshape(-1, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS)/255.
X_train, y_train = shuffle(X_train, y_train, seed=42)
#X_train, y_train, X_val, y_val = split(X_train, y_train, SPLIT_RATE)

#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Create a training graph that receives a batch of images and their respective labels and run a      #
#         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
#         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	print(X.shape)

	with tf.variable_scope('encoder'):
		out = tf.layers.conv2d(X, 4, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='same')
		print(out.shape)
		out = tf.layers.conv2d(out, 16, (3, 3), (1, 1), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='same')
		print(out.shape)
	with tf.variable_scope('decoder'):
		out = tf.layers.conv2d_transpose(out, 4, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print(out.shape)
		out = tf.layers.conv2d_transpose(out, 1, (3, 3), (2, 2), padding='same', activation=tf.nn.relu)
		print(out.shape)

	decoder_variables = [v for v in tf.global_variables() if v.name.startswith('decoder')]
	encoder_variables = [v for v in tf.global_variables() if v.name.startswith('encoder')]

	print encoder_variables, '\n\n\n\n'
	print(decoder_variables)

	loss = tf.reduce_mean(tf.reduce_sum((out-X)**2))

	encoder_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=encoder_variables)
	decoder_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=decoder_variables)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Run one training epoch using images in X_train and labels in y_train.                              #
# ---------------------------------------------------------------------------------------------------------- #
def training_epoch(session, lr):
	batch_list = np.random.permutation(len(X_train))

	start = time.time()
	train_loss1 = 0
	train_loss2 = 0
	for j in range(0, len(X_train), BATCH_SIZE):
		if j+BATCH_SIZE > len(X_train):
			break
		X_batch = X_train.take(batch_list[j:j+BATCH_SIZE], axis=0)

		ret1 = session.run([encoder_train_op, loss], feed_dict = {X: X_batch, learning_rate: lr, is_training: True})
		ret2 = session.run([decoder_train_op, loss], feed_dict = {X: X_batch, learning_rate: lr, is_training: True})
		train_loss1 += ret1[1]*BATCH_SIZE
		train_loss2 += ret2[1]*BATCH_SIZE

	pass_size = (len(X_train)-len(X_train)%BATCH_SIZE)
	print('Training Epoch:'+str(epoch)+' LR:'+str(lr)+' Time:'+str(time.time()-start)+' Loss1:'+str(train_loss1/pass_size)+' Loss2:'+str(train_loss2/pass_size))

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Evaluate images in Xv with labels in yv.                                                           #
# ---------------------------------------------------------------------------------------------------------- #
def evaluation(session, Xv, yv, name='Evaluation'):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE], is_training: False})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print(name+' Epoch:'+str(epoch)+' Time:'+str(time.time()-start)+' ACC:'+str(eval_acc/len(Xv))+' Loss:'+str(eval_loss/len(Xv)))

	return eval_acc/len(Xv), eval_loss/len(Xv)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Training loop.                                                                                     #
# ---------------------------------------------------------------------------------------------------------- #
NUM_EPOCHS_FULL = 200
S_LEARNING_RATE_FULL = 0.001
F_LEARNING_RATE_FULL = 0.001
BATCH_SIZE = 64

with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	# full optimization
	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, lr)

		#val_acc, val_loss = evaluation(session, X_val, y_val, name='Validation')

		cv2.imshow('input', X_train[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		rec = session.run(out, feed_dict = {X: X_train[0:0+1], is_training: False})
		cv2.imshow('output', rec[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
		cv2.waitKey(0)
