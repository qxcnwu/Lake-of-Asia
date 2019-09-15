import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os,shutil
import random
from PIL import Image

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('conv1'):
        w1=tf.get_variable('weights',[5,5,3,64],initializer=tf.truncated_normal_initializer(stddev=(5e-2)))
        b1=tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,w1,[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,b1))
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    with tf.variable_scope('conv2') as scope:
        w2 = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=(5e-2)))
        b2 = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(conv1, w2, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [1, -1])
        dim = reshape.get_shape()[1].value
        w3=tf.get_variable('weights', [dim,384], initializer=tf.truncated_normal_initializer(stddev=(0.04)))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(w3))
        b3 = tf.get_variable('bias', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, w3) + b3, name=scope.name)
    with tf.variable_scope('local4') as scope:
        w4 = tf.get_variable('weights', [384, 192], initializer=tf.truncated_normal_initializer(stddev=(0.04)))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(w4))
        b4 = tf.get_variable('bias', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, w4) + b4, name=scope.name)
        if train:
            local4=tf.nn.dropout(local4,0.5)
    with tf.variable_scope('local5') as scope:
        w5 = tf.get_variable('weights', [192, 96], initializer=tf.truncated_normal_initializer(stddev=(0.04)))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(w5))
        b5 = tf.get_variable('bias', [96], initializer=tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4, w5) + b5, name=scope.name)
        if train:
            local5=tf.nn.dropout(local5,0.5)
    with tf.variable_scope('softmax_linear') as scope:
        w6 = tf.get_variable('weights', [96,2], initializer=tf.truncated_normal_initializer(stddev=(1/96.0)))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(w6))
        b6 = tf.get_variable('bias', [2], initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local5, w6), b6, name=scope.name)
    return softmax_linear
def make_crood(jpg_path):
    list_make=[]
    Dir = os.listdir(jpg_path)
    num = len(Dir)
    num2=num%50
    num3=int(num/50)
    for i in range(num3):
        print('Trainning STEP:[{0}/{1}]'.format(i,num3))
        list=image_estimate(i*50,i*50+50)
        list_make+=list
    print('Trainning STEP:[{0}/{1}]'.format(i,num3))
    list=image_estimate(i*50,i*50+num2)
    list_make += list
    return list_make
def image_estimate(num_start,num_end):
    path=curr_dir+'/file/model/model.ckpt-9700'
    tf.reset_default_graph()
    list=[]
    x = tf.placeholder(tf.float32, [1, 32, 32, 3], name='x-input')
    y = inference(x, False, None)
    prediction = tf.argmax(y, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        flag=0
        for i in range(num_start,num_end):
            print("\rdeciding:[{0:.3}%][{1}/{2}]".format(flag / 50 * 100, flag, 50), end="")
            flag+=1
            imgpath = cut_dir + str(i + 1) + '.jpg'
            img = Image.open(imgpath)
            float_image = tf.image.per_image_standardization(img)
            pic = tf.expand_dims(float_image, 0)
            tt=pic.eval()
            predict =sess.run(prediction,feed_dict={x: tt})
            list.append(predict[0])
        coord.request_stop()
    return list
def tif_to_jpg(tif_path,jpg_path):
    im=Image.open(tif_path)
    im.save(jpg_path)
    print('success turn file')
def jpg_cut_image(H,W,jpg_path,cut_dir):
    name=1
    for i in range(H):
        for j in range(W):
            print("\rcutting:[{0:.3}%]".format(name/H/W*100),end="")
            save_path=os.path.join(cut_dir+str(name)+'.jpg')
            shutil.copy(jpg_path,save_path)
            im=Image.open(save_path)
            im=im.crop((i*32,j*32,(i+1)*32,(j+1)*32))
            im.save(save_path)
            name+=1
    print('\nsuccessful cut image')
    return 0
def make_pic(list,H,W,pic_path):
    x_num=0
    im=Image.open(pic_path)
    flag=0
    pixTuple = (255, 0, 255, 50)
    for i in range(H):
        for j in range(W):
            print("\rmaking:[{0:.3}%]".format(flag/ H / W * 100),end="")
            if int(list[flag])==0:
                for x in range(i * 32, (i + 1) * 32):
                    for y in range(j * 32, (j + 1) * 32):
                        im.putpixel((x, y), pixTuple)
            else:
                x_num+=1
            flag += 1
    im.save(pic_path)
    print('Get result success')
    return x_num
def hot_pixel_clear(list,H,W):
    data=np.zeros((H,W))
    flag=0
    for i in range(H):
        for j in range(W):
            data[i][j]=list[flag]
            flag+=1
    for i in range(H-1):
        for j in range(W):
            k=0
            try:
                if data[i-1][j]!=data[i][j]:
                    k+=1
                if data[i+1][j]!=data[i][j]:
                    k+=1
                if data[i][j+1]!=data[i][j]:
                    k+=1
                if data[i][j-1]!=data[i][j]:
                    k+=1
                if k>=3:
                    if data[i][j]==0:
                        data[i][j]=1
                    else:
                        data[i][j]=0
            except:
                pass
    i = H - 1
    for j in range(W):
        if data[i][j] != data[i - 1][j]:
            if data[i][j] == 0:
                data[i][j] = 1
            else:
                data[i][j] = 0
    list_out=[]
    for i in range(H):
        for j in range(W):
            list_out.append(int(data[i][j]))
    print('Success clear hot pixel')
    return list_out
def read_list(list_path):
    fd=open(list_path)
    lines=fd.readlines()
    list1=[]
    for line in lines:
        list1.append(line.strip())
    print('successly read list')
    return list1
def write_list(list,list_path):
    fd = open(list_path, 'w')
    for key in list:
        fd.write(str(key)+'\n')
    print('successful write list')
    return 0
def remove_file(path):
    Dir=os.listdir(path)
    for dir in Dir:
        file=os.path.join(path,dir)
        os.remove(file)
    print('successful remove files')
def np_list(list_path,H,W):
    list1=read_list(list_path)
    data=np.zeros((H,W))
    flag=0
    for i in range(H):
        for j in range(W):
            if list1[flag]=='0':
                data[i][j] =1
            else:
                data[i][j] =0
            flag+=1
    data=data.T
    print('successly make data')
    return data
def np_Tran(data,W,H):
    data2=np.zeros((W,H))
    for i in range(W):
        for j in range(H):
            data2[i][j]=data[H-j-1][i]
    print('successful turn data')
    return data2
def edge_maker(pic_path,label_path,pic_save_path,H,W):
    im=Image.open(pic_path)
    data=np_list(label_path,H,W)
    W,H=data.shape
    flag=0
    pixTuple = (255, 0, 255, 50)
    for i in range(W-1):
        for j in range(H-1):
            print("\rmake edge...ing:[{0:.3}%]".format(flag / H / W * 100), end="")
            flag+=1
            if data[i][j]!=data[i][j+1]:
                for x in range(i * 32, (i + 1) * 32):
                    for y in range(j*32+24,(j+1)*32):
                        im.putpixel((y,x), pixTuple)
            if data[i][j]!=data[i+1][j]:
                try:
                    for x in range(i * 32 + 24, (i + 1) * 32):
                        for y in range(j * 32, (j + 1) * 32):
                            im.putpixel((y, x), pixTuple)
                except:
                    pass
            else:
                pass
    im.save(pic_save_path)
    print('successul make edge')
    return 0
def jpg_cut_shape(jpg_path):
    im=Image.open(jpg_path)
    heigh,weigh=im.size
    H=heigh
    W=weigh
    im=im.resize((H,W))
    im.save(jpg_path)
    print('success reshape file')
    return int(H/32),int(W/32)
def jpg_cutting(jpg_path):
    im = Image.open(jpg_path)
    heigh, weigh = im.size
    H = (int(heigh / 800) + 1) * 800
    W = (int(weigh / 32) + 1) * 32
    im = im.resize((H, W))
    for i in range(int(H/800)):
        save_path=curr_dir+'/file/jpgdir/'+str(i)+'.jpg'
        k=im.crop((i*800,0,(i+1)*800,W))
        k.save(save_path)
    print('success reshape file')
    return i,heigh,weigh
def jpg_merge(jpg_path,num,W):
    img=Image.new('RGB',(num*800,W))
    for i in range(num):
        im=Image.open(curr_dir+'/file/jpgdir/'+str(i)+'.jpg')
        img.paste(im,(i*800,0,i*800+800,W))
    img.save(jpg_path)
    print('Successly merge pic')
    return num*800,W
def jpg_edge_merge(jpg_path,num,W):
    img = Image.new('RGB', (num * 800, W))
    for i in range(num):
        im = Image.open(curr_dir+'/file/jpgdir/' + str(i) + '3.jpg')
        img.paste(im, (i * 800, 0, i * 800 + 800, W))
    img.save(jpg_path)
    print('Successly merge pic edge')
    return 0
def area_num(heigh,weigh,x,y,total,unit):
    return 32*32*total*heigh*weigh*unit*unit/x/y

curr_dir = os.path.dirname(os.path.realpath(__file__))
tif_path=curr_dir+'/file/1.tif'
jpg_path=curr_dir+'/file/1.jpg'
jpg_path_done=curr_dir+'/file/done.jpg'
jpg_path_edge=curr_dir+'/file/edge.jpg'
answer_path=curr_dir+'/file/answer.txt'

if __name__ == "__main__":
    tif_to_jpg(tif_path, jpg_path)
    num,heigh,weigh=jpg_cutting(jpg_path)
    total=0
    unit=input('Please input unit:')
    unit=float(unit)
    for i in range(num+1):
        jpg_path = curr_dir+'/file/jpgdir/'+str(i)+'.jpg'
        pic_path = curr_dir+'/file/jpgdir/'+str(i)+'r.jpg'
        cut_dir = curr_dir+'/file/jpg/'
        list_path = curr_dir+'/file/jpgdir/'+str(i)+'.txt'
        pic_save_path =curr_dir+'/file/jpgdir/'+str(i)+'3.jpg'
        H, W = jpg_cut_shape(jpg_path)
        shutil.copyfile(jpg_path, pic_path)
        jpg_cut_image(H, W, jpg_path, cut_dir)
        list = make_crood(cut_dir)
        write_list(list, list_path)
        list = read_list(list_path)
        list=hot_pixel_clear(list, H, W)
        write_list(list, list_path)
        list = read_list(list_path)
        x_num=make_pic(list, H, W, jpg_path)
        total+=x_num
        edge_maker(pic_path, list_path, pic_save_path, H, W)
        remove_file(cut_dir)
    x,y=jpg_merge(jpg_path_done,num+1,W*32)
    jpg_edge_merge(jpg_path_edge, num+1, W*32)
    a=area_num(heigh,weigh, x, y, total, unit)
    path=curr_dir+'/file/jpgdir/'
    remove_file(path)
    fd=open(answer_path,'w')
    fd.write('Lake area:'+str(a)+'m*m')
    fd.close()
    print('Work done')
    print('The measure of area:',a,' m*m')
