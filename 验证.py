import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle
import math

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

def data_clean(data):
    data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    return data

def re_data(data,a,b):
    return data*a+b

def conside(x_data):
    return -0.8490372*x_data[0]+-1.3073596*x_data[1]+0.29399794*x_data[2]+0.68540055*x_data[3]+-0.23304984*x_data[4]+0.13862309*x_data[5]+ 0.3177481*x_data[6]+0.10823247*x_data[7]+-1.3524691e-05

x_data,y_data=read_file("C:/file/1.csv")
a=np.std(y_data,axis=0)
b=np.mean(y_data,axis=0)
x_data=data_clean(x_data)

answer_list=[]
year_list=[1985,1990,1995,2000,2005,2010,2015]
for i in range(7):
    answer_list.append(re_data(conside(x_data[i]),a,b))
    print('咸海{0}年实测面积：{1}m*m 经验公式计算面积：{2}m*m 绝对误差：{3}%'.format(year_list[i],y_data[i],re_data(conside(x_data[i]),a,b),abs(re_data(conside(x_data[i]),a,b)-y_data[i])/y_data[i]*100))