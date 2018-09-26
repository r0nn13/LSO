import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

class Dataset(object):
	def __init__(self):
		tmp = np.load('/home/ronnie/Downloads/depth_v2/depth_v2.npz')
		self.data = tmp['arr_0'].astype(np.float)/1000.0
		print(np.max(self.data))
		self.data = 2.0/(2.0+self.data)
		self._index = 0
		
	def next_batch(self,batch_size=128):
		self._index = self._index + batch_size
		self._index = self._index % np.shape(self.data)[0]
		
		return self.data[self._index:self._index+batch_size,:,:]

suncg = Dataset()

tf.reset_default_graph()

batch_size = 128

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 60, 80], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 60, 80], name='Y')

Y_flat = tf.reshape(Y, shape=[-1, 60 * 80])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 64

reshaped_dim = [-1, 15, 20, dec_in_channels]
inputs_decoder = 20*15 * dec_in_channels / 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))
	
def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 60, 80, 1])
        x = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)          
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd
		
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=1, padding='same', activation=None)
        #x = tf.layers.batch_normalization(x,training=True)
        x = lrelu(x)
        x0 = tf.depth_to_space(x, 4)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='same', activation=None)
        #x = tf.layers.batch_normalization(x,training=True)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='same', activation=None)
        #x = tf.layers.batch_normalization(x,training=True)
        x = lrelu(x)
        x = tf.depth_to_space(x, 2)
        x1 = tf.depth_to_space(x, 2)
        
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=None)
        #x = tf.layers.batch_normalization(x,training=True)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=None)
        #x = tf.layers.batch_normalization(x,training=True)
        x = lrelu(x)
        x = tf.depth_to_space(x, 2)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(tf.concat([x0,x],3), filters=32, kernel_size=4, strides=1, padding='same', activation=None)
        x = tf.layers.conv2d(x, filters=32, kernel_size=2, strides=1, padding='same', activation=lrelu)
        x = tf.layers.conv2d(x, filters=1, kernel_size=2, strides=1, padding='same', activation=lrelu)
        
        #x = tf.contrib.layers.flatten(x)
        #x = tf.layers.dense(x, units=60*80, activation=tf.nn.relu6)/6.0
        img = tf.reshape(2.0/(2.0+tf.abs(x)), shape=[-1, 60, 80])
        return img

sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 60*80])

dy,dx = tf.image.image_gradients(tf.expand_dims(2.0/dec-2.0,3)*49.2)
normal = tf.nn.l2_normalize(tf.concat([-dy,-dx,tf.ones_like(dx)],3),3)
dy_,dx_ = tf.image.image_gradients(tf.expand_dims(2.0/Y-2.0,3)*49.2)
normal_ = tf.nn.l2_normalize(tf.concat([-dy_,-dx_,tf.ones_like(dx)],3),3)

norm_loss = tf.reduce_sum(-tf.reduce_sum(tf.multiply(normal,normal_),3),axis=[1,2])
img_loss = tf.reduce_sum(tf.square(unreshaped - Y_flat), 1)

latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + 0.01*norm_loss + latent_loss )
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()
for e in range(30):
    for i in range(4500):
        batch = suncg.next_batch(batch_size=batch_size)
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            
        if not i % 500:
            ls, d, i_ls, d_ls, mu, sigm,n,n_= sess.run([loss, dec, img_loss, latent_loss, mn, sd,normal,normal_], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            cv2.imshow('win1',np.expand_dims(batch[0],2))
            cv2.imshow('win2',np.expand_dims(d[0],2))
            cv2.imshow('win3',n[0]*0.5+0.5)
            cv2.imshow('win4',n_[0]*0.5+0.5)
            cv2.waitKey(10)
            elapsed_time = time.time() - start_time
            print('{} - {}, total: {} image: {}, latent:{}'.format(i, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), ls, np.mean(i_ls), np.mean(d_ls)))
            save_path = saver.save(sess, "./model.ckpt")
		
