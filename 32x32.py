import tensorflow as tf
import numpy as np
import math
import os
import re

# # waring del
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        tf.histogram_summary(name,var)
        mean=tf.reduce_mean(var)
        tf.scalar_summary('mean/'+name,mean)
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.scalar_summary('stddev/'+name,stddev)
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
        reshape = tf.reshape(pool2, [batch_size, -1])
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
def make_batch(path):
    files = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解析读取的样例。
    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    labels = tf.cast(features['label'], tf.int64)
    images = tf.reshape(retyped_images, [32,32,3])
    float_image = tf.image.per_image_standardization(images)
    float_image.set_shape([32, 32, 3])
    return float_image,labels
# def _generate_image_and_label_batch(image, label, min_queue_examples,batch_size, shuffle):
#     num_preprocess_threads = 16
#     if shuffle:
#         images, label_batch = tf.train.shuffle_batch(
#             [image, label],
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             capacity=min_queue_examples + 3 * batch_size,
#             min_after_dequeue=min_queue_examples)
#     else:
#         images, label_batch = tf.train.batch(
#             [image, label],
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             capacity=min_queue_examples + 3 * batch_size)
#     tf.summary.image('images', images)
#     return images, tf.reshape(label_batch, [batch_size])
#
# def distorted_inputs_train(path, batch_size):
#     NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=7000
#     files = tf.train.match_filenames_once(path)
#     filename_queue = tf.train.string_input_producer(files, shuffle=True)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # 解析读取的样例。
#     features = tf.parse_single_example(serialized_example, features={
#         'image_raw': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.int64)
#     })
#     decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
#     retyped_images = tf.cast(decoded_images, tf.float32)
#     labels = tf.cast(features['label'], tf.int64)
#     images = tf.reshape(retyped_images, [32, 32, 3])
#     # IMAGE_SIZE=24
#     # height = IMAGE_SIZE
#     # width = IMAGE_SIZE
#     # distorted_image = tf.random_crop(images, [height, width, 3])
#     # distorted_image = tf.image.random_flip_left_right(distorted_image)
#     # distorted_image = tf.image.random_brightness(distorted_image,
#     #                                              max_delta=63)
#     # distorted_image = tf.image.random_contrast(distorted_image,
#     #                                            lower=0.2, upper=1.8)
#     float_image = tf.image.per_image_standardization(images)
#     float_image.set_shape([32, 32, 3])
#     min_fraction_of_examples_in_queue = 0.4
#     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
#                              min_fraction_of_examples_in_queue)
#     return _generate_image_and_label_batch(float_image,labels,
#                                            min_queue_examples, batch_size,
#                                            shuffle=True)
#
# def distorted_inputs_test(path, batch_size):
#     NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=2000
#     files = tf.train.match_filenames_once(path)
#     filename_queue = tf.train.string_input_producer(files, shuffle=True)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # 解析读取的样例。
#     features = tf.parse_single_example(serialized_example, features={
#         'image_raw': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.int64)
#     })
#     decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
#     retyped_images = tf.cast(decoded_images, tf.float32)
#     labels = tf.cast(features['label'], tf.int64)
#     images = tf.reshape(retyped_images, [32, 32, 3])
#     # IMAGE_SIZE=24
#     # height = IMAGE_SIZE
#     # width = IMAGE_SIZE
#     # distorted_image = tf.random_crop(images, [height, width, 3])
#     # distorted_image = tf.image.random_flip_left_right(distorted_image)
#     # distorted_image = tf.image.random_brightness(distorted_image,
#     #                                              max_delta=63)
#     # distorted_image = tf.image.random_contrast(distorted_image,
#     #                                            lower=0.2, upper=1.8)
#     float_image = tf.image.per_image_standardization(images)
#     float_image.set_shape([32, 32, 3])
#     min_fraction_of_examples_in_queue = 0.4
#     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
#                              min_fraction_of_examples_in_queue)
#     return _generate_image_and_label_batch(float_image,labels,
#                                            min_queue_examples, batch_size,
#                                            shuffle=True)
log_path='C:/file/git'
# # 模型参数
NUMBER_OF_EXAPLE=7000
BATCH_SIZE =200
batch_size=200
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.000001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.9999
num_preprocess_threads = 16
# # 训练batch提取
min_after_dequeue = 16
capacity = min_after_dequeue + 3 * BATCH_SIZE
images,labels=make_batch("C:/file/git/train.tfrecord")
images_batch, labels_batch = tf.train.shuffle_batch([images, labels], batch_size=200,num_threads=num_preprocess_threads,capacity=capacity,min_after_dequeue=min_after_dequeue)
images2,labels2=make_batch("C:/file/git/test.tfrecord")
images_batch1, labels_batch1 = tf.train.shuffle_batch([images2, labels2], batch_size=200,num_threads=num_preprocess_threads,capacity=capacity,min_after_dequeue=min_after_dequeue)
# 定义输出为4维矩阵的
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='x-input')
    y_ = tf.placeholder(tf.int64, [batch_size], name='y-input')
tf.summary.image('input',x,10)
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
y = inference(x,True,regularizer)
global_step = tf.Variable(0, trainable=False)
# #学习率定义
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))
tf.summary.scalar('cross entrop',loss)
# 设置指数衰减的学习率。
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,NUMBER_OF_EXAPLE / BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)
# 优化损失函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
# 反向传播更新参数和更新每一个参数的滑动平均值
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
#计算正确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        c=tf.argmax(y,1)
        correct_prediction = tf.equal(c, y_)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()
# 初始化TensorFlow持久化类。
saver = tf.train.Saver()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # #循环训练
    for i in range(TRAINING_STEPS):
        img, label,img2, label2= sess.run([images_batch,labels_batch,images_batch1, labels_batch1])
        summary,_,loss_value, step = sess.run([merged,train_op,loss,global_step], feed_dict={x: img, y_: label})
        dec=c.eval(feed_dict={x: img2, y_: label2})
        print(dec,label2)
        validate_acc = sess.run(accuracy, feed_dict={x: img2, y_: label2})
        writer.add_summary(summary,i)
        print("After %d training steps,loss : %g,acc: %g." % (step, loss_value,validate_acc))
        if i%100==0:
            saver.save(sess, "C:/file/git/model.ckpt", global_step=i)
    saver.save(sess, "C:/file/git/model.ckpt", global_step=i)
    writer.close()


