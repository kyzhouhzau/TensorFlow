#! usr/bin/env python3
# -*- coding:utf-8 -*-
'''
BP神经网络，是一种按照误差逆向传播算法训练的多层前馈神经网络

激活函数是用来加入非线性因素的，因为线性模型的表达力不够
'''
import tensorflow as tf
import numpy as np

#定义隐含层函数，传入参数：输入数据，输入数据尺寸，输出数据，输出数据尺寸，激活函数
def addLayer(inputData,inSize,outSize,activity_function=None):
    weights=tf.Variable(tf.random_normal([inSize,outSize]))
    basis=tf.Variable(tf.zeros([1,outSize])+0.1)
    weights_plus_b=tf.matmul(inputData,weights)+basis
    if activity_function is None:
        ans=weights_plus_b
    else:
        ans=activity_function(weights_plus_b)
    return ans


x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+0.5+noise



xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])



L1=addLayer(xs,1,10,activity_function=tf.nn.relu)
L2=addLayer(L1,10,1,activity_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square((ys-L2)),
                                  reduction_indices=[1]))


train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


for i in range(10000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))