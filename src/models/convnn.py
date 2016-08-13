import click
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def run(train_data, test_data, train=True, predict=True):
    epochs = 20000
    train_val_ratio = 0.7
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(
        open(os.path.join(project_dir, 'SETTINGS.json')).read()
    )
        

    # Imput layer
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling 1
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolutional layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Pooling 2
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully conected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout and softmax
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    if train:
        train_data_size = len(train_data)
        train_set = train_data[1001:]
        val_set = train_data[:1000]

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
        )
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())

        last_time = 0
        for i in range(epochs):
            batch = train_set.sample(50)
            batch_xs = batch.filter(like='pixel', axis=1).as_matrix() / 255.0
            batch_ys = batch.filter(like='dig', axis=1).as_matrix()
            if i % 100 == 0:
                train_eval = accuracy.eval(
                    feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}
                )
                remaining_time = (epochs - i) * (time.time() - last_time) / 6000
                last_time = time.time()
                print("step %d (%i min), training accuracy %g" % (i, remaining_time, train_eval))                
            train_step.run(
                feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}
            )
            if 1 % 1000 == 0:
                model_path = os.path.join(project_dir, settings['MODEL_PATH'], "conv_nn.ckpt")
                saver.save(sess, model_path)

        val_xs = val_set.filter(like='pixel', axis=1).as_matrix() / 255.0
        val_ys = val_set.filter(like='dig', axis=1).as_matrix()
        val_eval = accuracy.eval(
            feed_dict={x: val_xs, y_: val_ys, keep_prob: 1}
        )
        print("Validation accuracy %g" % (val_eval))
        model_path = os.path.join(project_dir, settings['MODEL_PATH'], "conv_nn.ckpt")
        saver.save(sess, model_path)
        sess.close()

    if predict:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from disk.
            model_path = os.path.join(project_dir, settings['MODEL_PATH'],
                                      "conv_nn.ckpt")
            saver.restore(sess, model_path)
            pred = []
            for dig in test_data.as_matrix() / 255.0:
                pred.append(sess.run(y_conv, feed_dict={x: [dig], keep_prob: 1.0}))
            # print(predict)
        pred = [[i + 1, np.argmax(one_hot_list)] for
                i, one_hot_list in enumerate(pred)]
        submission = pd.DataFrame(pred, columns=['ImageId', 'Label'])
        submission_path = os.path.join(project_dir, settings['SUBMISSION_PATH'],
                                       'submission_convnn.csv')
        submission.to_csv(submission_path, index=False)