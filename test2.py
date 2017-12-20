#! usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


matrix1 = tf.constant([[2,3]])
matrix2 = tf.constant([[2],
                      [3]])


product = tf.matmul(matrix1,matrix2)#matrix multipy 同np.dot(x1,x1


#method1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


