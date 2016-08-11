Kaggle - Digit Recognizer
=========================

Solutions for Kaggle's Digit Recognizer competition.

> The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is. As the competition progresses, we will release tutorials which explain different machine learning algorithms and help you to get started.

> The data for this competition were taken from the MNIST dataset. The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic within the Machine Learning community that has been extensively studied. More detail about the dataset, including Machine Learning algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html.


For further information check the [documentation](http://kaggle-mnist.readthedocs.io/en/latest/). 


System Specs
------------

The hardware / OS platform used to train the model:

* **OS:** Ubuntu 16.04 LTE 64-bit
* **Processor:** Intel Core i5-2450M CPU @ 2.50GHz x 4
* **Graphics:** Intel Sandybridge Mobile 


Requirements
------------

General Requirements:

* Numpy
* Pandas
* TensorFlow
* Click

Notebook requirements:

* Jupyter
* Matplotlib

Check [here](http://kaggle-mnist.readthedocs.io/en/latest/getting-started.html) for information on how to set up the enviroment 


Data Preparation
----------------

Download train and test data and edit SETTINGS.json or run:

    $ make data


Training
--------

To train a model just run:

    $ make train


Predicting
----------

To run the model on test datajust run:

    $ make predict