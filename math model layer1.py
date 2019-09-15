import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle

def read_file(path):
    fd=open(path)
    lines=fd.readlines()
    num=len(lines)
    x_data=np.zeros((num,8))
    y_data=np.zeros((num,1))
    flag=0
    for line in lines:
        for i in range(8):
            x_data[flag][i]=line.split(',')[3+i]
        y_data[flag][0]=line.split(',')[2]
        # y_data[flag][1]=line.split(',')[-1]
        flag+=1
    return x_data,y_data

def Z_ScoreNormalization(x,mu,sigma):
    return (x-mu)/sigma

def data_clean(data):
    data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    return data

def inferance(IN,w1,b1):
    return tf.matmul(IN,w1) + b1

IN=8
OUT=1
LAYER1_NODE = 1
REGULARAZTION_RATE = 0.0001
Learning_Rate=0.0001
Epoch=10000

log_path='C:/file/al'

with tf.name_scope('input'):
    in_put=tf.placeholder(tf.float32,[None,IN],name='in_put')
    out_put=tf.placeholder(tf.float32,[None,OUT],name='out_put')

with tf.name_scope('model'):
    w1 = tf.Variable(tf.truncated_normal([IN, LAYER1_NODE], stddev=1.0),name='w1')
    b1 = tf.Variable(tf.constant(1.0, shape=[LAYER1_NODE]),name='b1')
    prediction=inferance(in_put,w1,b1)

with tf.name_scope('loss_func'):
    cross_entropy_mean = tf.reduce_mean(tf.square(out_put-prediction))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(w1)
    Loss = cross_entropy_mean + regularaztion

Optimizer=tf.train.GradientDescentOptimizer(Learning_Rate).minimize(Loss)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    x_data,y_data=read_file('C:/file/1.csv')
    x_data=data_clean(x_data)
    y_data=data_clean(y_data)
    f=open('C:/file/wb_loss.txt','w')
    k=open('C:/file/loss.txt','w')
    for i in range(Epoch):
        loss_sum=0.0
        for In,out in zip(x_data,y_data):
            In=In.reshape(1,8)
            out=out.reshape(1,1)
            _,loss=sess.run([Optimizer,Loss],feed_dict={in_put:In,out_put:out})
            loss_sum+=loss
        x_data,y_data=shuffle(x_data,y_data)
        loss_avg=loss_sum/len(y_data)
        print('epoch=', i + 1, 'loss=', loss_avg)
        if i%100==0:
            k.write(str(loss_avg)+'\n')
    w = sess.run(w1)
    b = sess.run(b1)
    f.write(str(w) + ',' + str(b) + ',' + str(loss_avg) + '\n')
    f.close()
    k.close()

