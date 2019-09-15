import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle

def read_file(path):
    fd=open(path)
    lines=fd.readlines()
    num=len(lines)
    x_data=np.zeros((num,8))
    y_data=np.zeros((num,2))
    flag=0
    for line in lines:
        for i in range(8):
            x_data[flag][i]=line.split(',')[3+i]
        y_data[flag][0]=line.split(',')[2]
        y_data[flag][1]=line.split(',')[-1]
        flag+=1
    return x_data,y_data

def Z_ScoreNormalization(x,mu,sigma):
    return (x-mu)/sigma

def data_clean(data):
    data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    return data

def inferance(IN,w1,b1,w2,b2,w3,b3,w4,b4):
    layer1 = tf.nn.relu(tf.matmul(IN,w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
    return tf.matmul(layer3, w4) + b4

IN=8
OUT=2
LAYER1_NODE = 8
LAYER2_NODE = 4
LAYER3_NODE = 2
REGULARAZTION_RATE = 0.0001
Learning_Rate=0.0001
Epoch=500000

in_put=tf.placeholder(tf.float32,[None,IN],name='in_put')
out_put=tf.placeholder(tf.float32,[None,OUT],name='out_put')

with tf.name_scope('model'):
    w1 = tf.Variable(tf.truncated_normal([IN, LAYER1_NODE], stddev=1.0),name='w1')
    b1 = tf.Variable(tf.constant(1.0, shape=[LAYER1_NODE]),name='b1')
    w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=1.0),name='w2')
    b2 = tf.Variable(tf.constant(1.0, shape=[LAYER2_NODE]),name='b2')
    w3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, LAYER3_NODE], stddev=1.0),name='w3')
    b3= tf.Variable(tf.constant(1.0, shape=[LAYER3_NODE]),name='b3')
    w4 = tf.Variable(tf.truncated_normal([LAYER3_NODE, OUT], stddev=1.0),name='w4')
    b4 = tf.Variable(tf.constant(1.0, shape=[OUT]),name='b4')
    prediction=inferance(in_put,w1,b1,w2,b2,w3,b3,w4,b4)

with tf.name_scope('loss_func'):
    cross_entropy_mean = tf.reduce_mean(tf.square(out_put-prediction))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(w1) + regularizer(w2)+regularizer(w3)+regularizer(w4)
    Loss = cross_entropy_mean + regularaztion

Optimizer=tf.train.GradientDescentOptimizer(Learning_Rate).minimize(Loss)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    x_data,y_data=read_file('C:/file/ZSB.txt')
    x_data=data_clean(x_data)
    y_data=data_clean(y_data)
    for i in range(Epoch):
        loss_sum=0.0
        for In,out in zip(x_data,y_data):
            In=In.reshape(1,8)
            out=out.reshape(1,2)
            _,loss=sess.run([Optimizer,Loss],feed_dict={in_put:In,out_put:out})
            loss_sum+=loss
        x_data,y_data=shuffle(x_data,y_data)
        loss_avg=loss_sum/len(y_data)
        print('epoch=',i+1,'loss=',loss_avg)

