import tensorflow as tf
import numpy as np
import cv2
import glob
import random

paths_train_t = glob.glob('/home/haneul/Crosswalk_Image/train/img/t*.jpg')
paths_train_f = glob.glob('/home/haneul/Crosswalk_Image/train/img/f*.jpg')
paths_test = glob.glob('/home/haneul/Crosswalk_Image/test/img/*.jpg')


def batch(t, size):
    paths = []

    if t == 'train':
        paths_t = random.sample(paths_train_t, size)
        paths_f = random.sample(paths_train_f, size)
        paths = random.sample(paths_t + paths_f, size)

    elif t == 'test':
        paths = random.sample(paths_test, size)

    retx, rety = [], []

    l = len(t)

    for path in paths:
        retx.append(cv2.imread(path).flatten().tolist())
        flag = int(path[34 + l:35 + l] == 't')
        rety.append([flag, 1-flag])

    retx, rety = np.array(retx), np.array(rety)

    return retx, rety


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

t = tf.placeholder(tf.float32, [None, 2])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
i = 0

for _ in range(20000):
    i += 1
    print('step_begin ', i)
    x_train, t_train = batch('train', 30)
    print('batch loaded')
    sess.run(train_step, feed_dict={x: x_train, t: t_train, keep_prob: 0.75})
    print('step_end ', i)

    x_test, t_test = batch('test', 30)
    loss_val, acc_val = sess.run([loss, accuracy],
                                 feed_dict={x: x_test, t: t_test, keep_prob: 1.0})
    print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

    if i % 500 == 0:
        saver.save(sess, 'cnn_session', global_step=i)
