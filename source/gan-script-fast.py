import tensorflow as tf
import datetime
from data import Dataset

from tensorflow.examples.tutorials.mnist import input_data

class Gan():
    # Define the discriminator network
    def __init__(self):
        img_shape = [None,28,28,1]
        #img_shape = [None,64,64,1]
        self.x_placeholder = tf.placeholder(tf.float32, shape=img_shape, name='x_placeholder')
        # x_placeholder is for feeding input images to the discriminator

    def discriminator(self, images, reuse_variables=None):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

            # Alterar valor de stddev para 0 (Por default, o construtor do método já inicializa com zero)
            d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02)) # numpy.random.normal -> verificar essa função, q é equivalente
            d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
            d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
            d1 = d1 + d_b1
            d1 = tf.nn.relu(d1)
            d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print("d1: {}".format(d1.shape))

            d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
            d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
            d2 = d2 + d_b2
            d2 = tf.nn.relu(d2)
            d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print("d2: {}".format(d2.shape))

            d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
            d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
            d3 = tf.matmul(d3, d_w3)
            d3 = d3 + d_b3
            d3 = tf.nn.relu(d3)
            print("d3: {}".format(d3.shape))

            d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
            self.d4 = tf.matmul(d3, d_w4) + d_b4
            print("d4: {}".format(self.d4.shape))
            return self.d4

    def generator(self, batch_size, z_dim):
        z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z') # passar por parâmetro ao invés de gerar aqui
        g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
        g1 = tf.nn.relu(g1)
        print("g1: {}".format(g1.shape))

        g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])
        print("g2: {}".format(g2.shape))

        g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])
        print("g3: {}".format(g3.shape))

        g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
        self.g4 = self.g4 + g_b4
        self.g4 = tf.sigmoid(self.g4)
        print("g4: {}".format(self.g4.shape))

        return self.g4
        # Dimensions of g4: batch_size x 28 x 28 x 1

    def train(self, dataset):
        z_dimensions = 100
        batch_size = 50

        Gz = self.generator(batch_size, z_dimensions)
        # Gz holds the generated images

        Dx = self.discriminator(self.x_placeholder)
        # Dx will hold discriminator prediction probabilities
        # for the real MNIST images

        Dg = self.discriminator(Gz, reuse_variables=True)
        # Dg will hold discriminator prediction probabilities for generated images

        # Define losses
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

        # Define variable lists
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

        # Define the optimizers
        # Train the discriminator
        d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
        d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

        # Train the generator
        g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

        # From this point forward, reuse variables
        tf.get_variable_scope().reuse_variables()

        sess = tf.Session() # Verificar pq não passa um grafo como parâmetro

        # Begin tensorboard area
        # Send summary statistics to TensorBoard
        tf.summary.scalar('Generator_loss', g_loss)
        tf.summary.scalar('Discriminator_loss_real', d_loss_real)
        tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

        images_for_tensorboard = self.generator(batch_size, z_dimensions)
        tf.summary.image('Generated_images', images_for_tensorboard, 5)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        # End tensorboard area

        sess.run(tf.global_variables_initializer())

        # Pre-train discriminator
        print("Pre-train discriminator")
        for i in range(300):
            real_image_batch = dataset.next_batch(50)
            #print("real_image_batch: ")
            #print(real_image_batch.shape)
            _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                                {self.x_placeholder: real_image_batch})
            
        # Train generator and discriminator together
        print("Train generator and discriminator together")
        for i in range(100000):
            real_image_batch = dataset.next_batch(50)
            #print(real_image_batch.shape)

            # Train discriminator on both real and fake images
            _, __ = sess.run([d_trainer_real, d_trainer_fake], {self.x_placeholder: real_image_batch})

            # Train generator
            _ = sess.run(g_trainer)

            if i % 10 == 0:
                # Update TensorBoard with summary statistics
                summary = sess.run(merged, {self.x_placeholder: real_image_batch})
                writer.add_summary(summary, i)

        saver = tf.train.Saver()
        path_model = 'pretrained-model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_gan.ckpt'
        saver.save(sess, path_model)
        print("The model has saved in: " + path_model)
        
if __name__ == "__main__":
    d = Dataset()
    _ = d.load_all_images('../data_part1/train', '../data_part1/test', height=28, width=28)

    print("Imagens carregadas!")
    net = Gan()
    print("Rede inicializada!")
    net.train(d)