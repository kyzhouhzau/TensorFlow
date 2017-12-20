#! usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,{xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)#产生正太分布均shape表示维数，mean 表示均值默认0，stddev表示标准差默认1
    return tf.Variable(inital)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return initial

def conv2d(x,W):
    #stride [1,x_movement,y_movememt,1]
    # Must have strides[0]=stride[3]=1
    #padding可以是'SAME'或者‘'VALID'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    #第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


keep_prob = tf.placeholder(tf.float32)#dropout
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(xs,[-1,28,28,1])#[-1,28,28,1]这里的-1指的是不管图片的维度后期加上，28指28*28，1指黑白色
# print(x_image.shape)#[n_sample,28,28,1]

## conv1 layer##

W_conv1 = weight_variable([5,5,1,32])#patch 5*5 ,insize（图像厚度） 1,outsize 32
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                        #output size 14*14*32

## conv2 layer##
W_conv2 = weight_variable([5,5,32,64])#patch 5*5 ,insize 32,outsize 64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                        #output size 7*7*64

#### conv3 layer##
# W_conv3 = weight_variable([5,5,64,128])#patch 5*5 ,insize 64,outsize 128
# b_conv3 = bias_variable([128])

# h_conv3= tf.nn.relu(conv2d(h_pool2,W_conv2)+b_conv3)#output size 7*7*128
# h_pool3 = max_pool_2x2(h_conv3)                      #output size 3*3*128

##func1 layer##
w_fc1 = weight_variable([7*7*64,1024])#7*7*64是第二层池化后的结果，，1024是随意制定的较大空间
b_fc1=bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#数据维数转换。。。[n_sample,7,7,64]->>[n_sample,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

##func2 layer##
w_fc2 = weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

##the error between prediction and real data##
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 50 ==0:
        print(computer_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))

save_path = saver.save(sess,r"C:\Users\zhouk\PycharmProjects\TensorFlow\tensorflow\nums\save_net.ckpt")