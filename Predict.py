import tensorflow as tf
import numpy as np
import cv2
import glob
import random

paths_t = glob.glob('/home/haneul/Crosswalk_Image (copy)/train/img/t*.jpg')
paths_f = glob.glob('/home/haneul/Crosswalk_Image (copy)/train/img/f*.jpg')
paths = random.sample(paths_f, 1000) + random.sample(paths_t, 1000)
random.shuffle(paths)

def load_image(path):
    img = cv2.resize(cv2.imread(path), (300, 300), cv2.INTER_AREA).flatten().tolist()

    return np.array(img)


num_filters1 = 32
num_filters2 = 64

x = tf.placeholder(tf.float32, [None, 300 * 300 * 3])
x_image = tf.reshape(x, [-1, 300, 300, 3])

with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.Variable(tf.truncated_normal([45, 45, 3, num_filters1], stddev=0.001))
    # print('W_conv1 : ', W_conv1.get_shape())
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')

    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
    h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 15, 15, 1], strides=[1, 15, 15, 1], padding='SAME', name='pool1')
norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
# print('h_pool1 : ', h_pool1.get_shape())

with tf.variable_scope('conv2') as scope:
    W_conv2 = tf.Variable(tf.truncated_normal([15, 15, num_filters1, num_filters2], stddev=0.001))
    # print('W_conv2 : ', W_conv2.get_shape())
    h_conv2 = tf.nn.conv2d(norm1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')

    b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
    h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

norm2 = tf.nn.lrn(h_conv2_cutoff, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name='pool2')
# print('h_pool2 : ', h_pool2.get_shape())
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * num_filters2])

num_units1 = 7 * 7 * num_filters2
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 2]))
b0 = tf.Variable(tf.zeros([2]))
p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './cnn_session_/-1700')

for path in paths:
    res = sess.run([p], feed_dict={x: [load_image(path)], keep_prob: 1.0})

    ans = ''

    if np.argmax(res[0], 1) == 0:
        ans = 'Crosswalk'

    else:
        ans = 'Not Crosswalk'

    img = cv2.resize(cv2.imread(path), (600, 600), interpolation=cv2.INTER_CUBIC)
    cv2.putText(img, ans, (0, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (60, 60, 255), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(750)
    if k == 27:
        cv2.destroyAllWindows()
        break

