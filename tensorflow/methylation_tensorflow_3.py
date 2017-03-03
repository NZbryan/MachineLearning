# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:08:01 2017

@author: yinyang_ni
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:35:20 2017

@author: yinyang_ni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:12:08 2017

@author: yinyang_ni
"""
import tensorflow as tf
import pandas as pd
import numpy as np

#------------------- input data
common_data1=pd.read_csv('D:/python_workspace/common_data1.csv',index_col=0)
#del 0
common_data1=common_data1.ix[:, (common_data1 != 0).any()]

#删除两个癌前病变
common_data2=common_data1.drop(['CR003-045T','CR003-045N'],axis=0)

# T，N
data_train = common_data2.loc[(common_data2['SAMPLE_TYPE']=='T') | (common_data2['SAMPLE_TYPE']=='N')]


common_2=data_train.T

n_rows=data_train.shape[0]
n_cols=data_train.shape[1]
X = np.array(data_train.ix[:,:n_cols-1])
y1 = data_train["SAMPLE_TYPE"]
y_test = np.array(y1)

from sklearn.preprocessing import label_binarize
y = label_binarize(y_test, classes=["N","T"])
j1=y.T[0]
j2 = 1-j1

y = np.c_[y,j2]

x_data = X.astype("float32")
x_data = X[:,:9600]
y_data = y
#==============================================================================
# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# y = lb.fit_transform(y_test)
#==============================================================================


#---------------------------------------------------------------

sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, x_data.shape[1]])
y_ = tf.placeholder("float", shape=[None, 2])


n_x_data = int(x_data.shape[1]/4)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

W_conv1 = weight_variable([1, 5, 1, 32])
b_conv1 = bias_variable([32])

x_input = tf.reshape(x, [-1,1,x_data.shape[1],1])

h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([n_x_data * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, n_x_data*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
#==============================================================================
# for i in range(1000):
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:x_data, y_: y, keep_prob: 1.0})
#     print ("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: x_data, y_: y, keep_prob: 0.5})
#==============================================================================
for i in range(100):   
    train_accuracy = accuracy.eval(feed_dict={
            x:x_data, y_: y, keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: x_data, y_: y, keep_prob: 1.0})