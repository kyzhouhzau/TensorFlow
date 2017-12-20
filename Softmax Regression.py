#! usr/bin/env python3
# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIS_data/",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()#将session 注册为默认的session不同的session 间的数据和运算是相互独立的。
x = tf.placeholder(tf.float32,[None,784])#Placeholder 是数据输入的地方。第一个参数是数据类型，第二个参数是tensor 的shape .
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#用tensorflow写Softmax Regression代码。
y = tf.nn.softmax(tf.matmul(x,w)+b)
#定义loss函数，对多分类问题，通常使用Cross-entropy
y_= tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#定义优化算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#全局参数初始化
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

    print("Train section %d" % i)



#以上为模型训练完成
#接下来对模型训练的准确率进行验证tf.argmax是从一个tensor中寻找最大的序号，tf.argmax(y,1)就是各个预测数字中
#最大的那个，而tf.argmax(y_,1)则是找样本的真实数字类别。
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
print(correct_prediction)
#统计全部样本预测的accura,先要用tfcast将之前的correct_prediction输出的bool值转换为float32
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy)
print(accuracy.eval({x:mnist.validation.images,y_:mnist.validation.labels}))




