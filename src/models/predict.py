import click
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
    

@click.command()
def main():
    #############
    # LOAD DATA #
    #############
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(
        open(os.path.join(project_dir, 'SETTINGS.json')).read()
    )

    test_path = os.path.join(project_dir, settings['TEST_DATA_PATH'])
    test_data = pd.read_csv(test_path)

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

    ######################
    # LOAD AND RUN MODEL #
    ######################
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
                                   'logistic_regression.csv')
    submission.to_csv(submission_path, index=False)

if __name__ == '__main__':
    main()