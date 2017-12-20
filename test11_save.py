#! usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
# #
# Save to file
#
# remember to define the same dtype and shape
# W = tf.Variable([[1,2,3],[2,3,4]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.int32,name='biases')
#
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
#
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(W.shape, b.shape)
#     save_path = saver.save(sess,r"C:\Users\zhouk\PycharmProjects\TensorFlow\tensorflow\nums\save_net.ckpt")
#     print("Save to path:",save_path)


#

W = tf.Variable(np.arange(32).reshape((5,5,1,32)),dtype=tf.float32)

b = tf.Variable(np.arange(10).reshape((10)),dtype=tf.float32)

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,r"C:\Users\zhouk\PycharmProjects\TensorFlow\tensorflow\nums\save_net.ckpt")
    print("weights",sess.run(W))
    print("biases", sess.run(b))




