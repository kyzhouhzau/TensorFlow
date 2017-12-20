#! usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#this is the data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


#hyperparameters

lr = 0.01
training_iters = 100000
batch_size = 128


n_input = 28#MNIST data input 每一行有28个picture
n_steps = 28#有28行
n_hidden_unis = 128
n_class = 10


#tf.Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_class])


#Define weights
weights = {
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_input,n_hidden_unis])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_class]))

}
biases = {
    #（128，)
    'in':tf.Variable(tf.constant(0.1,shape = [n_hidden_unis],)),
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_class,]))

}




def RNN(X,weights,biases):
#hiddenlayer for input to cell
#############################################
#X(128batch,28steps,28inputs)
#==>(128*128,28 inputs)
    X = tf.reshape(X,[-1,n_input])
#X_in==>(128batch*28 steps,128 hidden)
    X_in = tf.matmul(X,weights['in'])+biases['in']
#X_in==>（128batch,28 steps,128hidden）
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])
##############################################
#cell
##############################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    #lstm cell is divided into two parts(c_state,m_state
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major = False)
##############################################
#hidden layer for out put as final results
###############################################
    results = tf.matmul(states[1],weights['out'])+biases['out']
    #
    # outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    # results = tf.matmul(outputs[-1],weights['out']) + biases['out']
    return results


pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_step = tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0

    while step*batch_size < training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_input])
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        if step % 20 ==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))

        step+=1










