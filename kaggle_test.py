#! usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import warnings
warnings.filterwarnings('ignore')
seed = 782
np.random.seed(seed)
data = pd .read_csv('./train.csv')
print(data.head())
train = data.as_matrix()
train_y  = train[:,0].astype('int8')
train_x = train[:,1:].astype('float64')
print('Shape train images:(%d,%d)'%train_x.shape)
print('Shape label shape:(%d)'%train_y.shape)
data1 = pd .read_csv('./test.csv')
print(data1.head())
test = data1.as_matrix().astype('float64')
print('Shape Test images:(%d,%d)'%test.shape)
images = data.iloc[:,1:].values
index_in_epoch = 0
train_images = images[2000:]
num_examples = train_images.shape[0]
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_x[start:end], train_y[start:end]
# def show_image(image,shape,label='',cmp=None):
#     img = np.reshape(image,shape)
#     plt.imshow(img,cmap=cmp,interpolation='none')
#     plt.title(label)
#     plt.figure(figsize=(12, 10))
#     x, y = 5, 10
#     for i in range(0, (y * x)):
#         plt.subplot(x, y, i + 1)
#         ni = np.random.randint(0, train_x.shape[0], 2)[1]
#
#         show_image(train_x[ni], (28, 28), train_y[ni], cmp='gray')
#     plt.show()
# def count_example_per_digit(examples):
#     hist = np.ones(10)
#     for y in examples:
#         hist[y]+=1
#     colors = []
#     for i in range(10):
#         colors.append(plt.get_cmap('viridis')(np.random.uniform(0.0,1.0,1)[0]))
#     bar = plt.bar(np.arange(10),hist,0.7,color=colors)
#     plt.grid()
#     plt.show()
# count_example_per_digit(train_y)
train_x /= 255
test /= 255
train_y = pd.get_dummies(train_y).as_matrix()
import tensorflow as tf
def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
image = tf.reshape(xs, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

# 全链接层
w_fc1 = weight_variable([7 * 7 * 64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_f1_drop = tf.nn.dropout(h_fc1, keep_prob)

wf_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_f1_drop, wf_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))

train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

correct_prodection = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prodection, 'float'))

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys =next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(sess.run(accuracy,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))
