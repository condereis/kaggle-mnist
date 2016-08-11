Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Data Preparation
^^^^^^^^^^^^^^^^

`make data` will download train.csv and test.csv files to /data/raw/


Model Training
^^^^^^^^^^^^^^

`make train` will train the model using train.csv and save to /models/


Making Predictions
^^^^^^^^^^^^^^^^^^

`make predict` will run the model on test.csv data and save a CVF file
with the predictions to /submissions/