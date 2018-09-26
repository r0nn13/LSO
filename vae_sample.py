import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.python.ops.parallel_for import pfor
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

flags = tf.app.flags
FLAGS = flags.FLAGS
print(flags.FLAGS.op_conversion_fallback_to_while_loop)
flags.FLAGS.op_conversion_fallback_to_while_loop = True
print(flags.FLAGS.op_conversion_fallback_to_while_loop)

dec_in_channels = 1
n_latent = 64

reshaped_dim = [-1, 15, 20, dec_in_channels]
inputs_decoder = 20*15 * dec_in_channels / 2

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 60, 80], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 60, 80], name='Y')

code = tf.placeholder(dtype=tf.float32, shape=[None, n_latent], name='code')

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
		
#sampled, mn, sd = encoder(X_in, keep_prob)
prox = decoder(code, keep_prob)

res = prox - Y
res_vec = tf.reshape(res,[1,-1])[:,:500]
prox0_vec = tf.reshape(prox,[1,-1,1])


J = gradients.batch_jacobian(res_vec,code, use_pfor=True)

#J = tf.reshape(J,[-1,80*60,64])
print(J)

jtj = tf.matmul(J,J,transpose_a=True)
jtjm1 = tf.matrix_inverse(jtj + 0.01*tf.eye(64))
jtr = tf.matmul(J,tf.expand_dims(res_vec,2),transpose_a=True)
Hm1r = tf.matmul(jtjm1,jtr)

print(Hm1r)

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True 

saver = tf.train.Saver()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver.restore(sess, "./model.ckpt")

code_np = np.zeros((1,64))

depth_gt = np.expand_dims(cv2.resize(cv2.imread('/home/ronnie/Downloads/depth_v2/0a9a73086e03cae699ead26a63d38581/000002_depth.png',cv2.IMREAD_UNCHANGED),(80,60))[:,:]/1000.0,0)
depth_gt = 2.0/(2.0+depth_gt)

with tf.device('/device:GPU:0'):  # Replace with device you are interested in
    bytes_in_use = BytesInUse()

for i in range(5):
    print(sess.run(bytes_in_use)/1000000)
    start_time = time.time()
    proxi_np,Hm1r_np = sess.run([prox,Hm1r],feed_dict={keep_prob:1.0, code: code_np, Y: depth_gt})
    end_time = time.time()
    print(end_time-start_time)
    code_np = code_np - Hm1r_np[:,:,0]
    print(np.shape(proxi_np[0]))
    cv2.imshow('win',np.expand_dims(proxi_np[0],2))
    cv2.imshow('win1',np.expand_dims(depth_gt[0],2))
    cv2.waitKey(200)

#code1 = np.random.randn(1,64)
#code2 = np.random.randn(1,64)

#for i in range(100):
#    alpha = float(i)/100.0
#    out = sess.run(prox,feed_dict={keep_prob:1.0, code: alpha*code1 + (1-alpha)*code2})
#    cv2.imshow('win',out[0])
#    cv2.waitKey(10)

