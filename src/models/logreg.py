import click
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import time


def run(train_data, test_data, train=True, predict=True):
    epochs = 1000
    train_val_ratio = 0.7
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(
        open(os.path.join(project_dir, 'SETTINGS.json')).read()
    )
    train_data_size = len(train_data)
    train_set = train_data[:int(train_data_size * train_val_ratio)]
    val_set = train_data[int(train_data_size * train_val_ratio) + 1:]

    ###########################
    # CREATE OPERATIONS GRAPH #
    ###########################
    # input tensor
    x = tf.placeholder(tf.float32, [None, 784])

    # weights (w) and biases (s)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # model output (y)
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # target and cross entropy
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )

    # train step and evaluation
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if train:
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)

        train_eval_list = []
        val_eval_list = []
        print('Training: Logistic Regression')
        start = time.time()
        for i in range(epochs):
            batch = train_set.sample(frac=0.1)
            batch_xs = batch.drop('label', axis=1).as_matrix() / 255.0
            batch_ys = pd.get_dummies(batch['label']).as_matrix()
            val_xs = val_set.drop('label', axis=1).as_matrix() / 255.0
            val_ys = pd.get_dummies(val_set['label']).as_matrix()

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            train_eval = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            val_eval = sess.run(accuracy, feed_dict={x: val_xs, y_: val_ys})

            train_eval_list.append(train_eval)
            val_eval_list.append(val_eval)
            done = int(i + 1)
            sys.stdout.write(
                "\r[%s%s] %s%%" %
                ('=' * done / epochs, ' ' * (100 - done) / epochs, done)
            )
            sys.stdout.flush()
        print()

        model_path = os.path.join(project_dir, settings['MODEL_PATH'],
                                  "logistic_regression.ckpt")
        saver.save(sess, model_path)
        sess.close()


    if predict:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from disk.
            model_path = os.path.join(project_dir, settings['MODEL_PATH'],
                                      "logistic_regression.ckpt")
            saver.restore(sess, model_path)
            predict = sess.run(y, feed_dict={x: test_data.as_matrix() / 255.0})

        pred = [[i + 1, np.argmax(one_hot_list)] for
                i, one_hot_list in enumerate(predict)]
        submission = pd.DataFrame(pred, columns=['ImageId', 'Label'])
        submission_path = os.path.join(project_dir, settings['SUBMISSION_PATH'],
                                       'submission_logreg.csv')
        submission.to_csv(submission_path, index=False)