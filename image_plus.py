import tensorflow as tf
def _generate_image_and_label_batch(image, label, min_queue_examples,batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])
def distorted_inputs(path, batch_size):
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=16000
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
    with tf.Session() as sess:
        print(labels.eval())
    images = tf.reshape(retyped_images, [32, 32, 3])
    IMAGE_SIZE=24
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    distorted_image = tf.random_crop(images, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, width, 3])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image,labels,
                                           min_queue_examples, batch_size,
                                           shuffle=True)
path='C:/file/city_coun/train.tfrecord'
batch_size=100
images,labels=distorted_inputs(path, batch_size)
with tf.Session() as sess:
    print(labels.eval())