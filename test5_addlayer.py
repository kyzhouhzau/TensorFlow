#! usr/bin/env python3
# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#定义隐含层函数
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    #tensorboard可视化，这一句是定义一个大的可视化内容名称，原因是下面还有一些小的需要可视化的变量。
    layer_name = 'layer%d' % n_layer
    with tf.name_scope(layer_name):
        #可视化Weights变量
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'\Weight', Weights)
        # 可视化biases变量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'\\biases',biases)
        # 可视化Wx_plus_b变量
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
            tf.summary.histogram(layer_name+'\Wx_plus_b',Wx_plus_b)
        with tf.name_scope('outputs'):
            if activation_function is None:
                outputs=Wx_plus_b
            else:
                outputs=activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'\outputs',outputs)
        return outputs

#Make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

# 可视化xs,以及ys变量
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,2,activation_function=None)

# 可视化loss变量
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-prediction)),
                      reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
# 可视化train_step变量

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
#合并所有需要可视化的条目并写入log文件夹中。
merged = tf.summary.merge_all()
write = tf.summary.FileWriter(r'C:\Users\zhouk\Desktop\logs',sess.graph)

#保存所有参数，也就是对最优模型进行保存。下次调用时可以直接导入进行新数据的预测。在最后需要saver.save()最终参数
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

#在图中画出训练数据的分布。
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 10==0:
    # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        #删除上一步产生的直线，否则会发生叠合。
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #在散点图上展示我们预测的那条直线
        predicton_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,predicton_value,'r-',lw=5)
        plt.pause(0.05)

        #写入tensorboard
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        write.add_summary(result,i)
#注意文件名以.ckpt结尾。
saver.save(sess,r'C:\Users\zhouk\PycharmProjects\TensorFlow\tensorflow\nums\save_net.ckpt')


################################################
#保存最终拟合结果
# plt.savefig(r'C:\Users\zhouk\PycharmProjects\TensorFlow\tensorflow\nums\result.png')
###############################################


###############################################
#可视化参见这里
#cd C:\Users\zhouk\Desktop文件夹运行下列命令
#tensorboard --logdir="logs\"
###############################################


###############################################
#保存的参数的利用途径
import tensorflow as tf



###############################################

