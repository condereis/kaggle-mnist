import click
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

import logreg
import convnn

epochs = 1000
train_val_ratio = 0.7

@click.command()
@click.option('--model', type=click.Choice(['logreg', 'convnn']), default='convnn')
def main(model):
    # Load data
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(open(os.path.join(project_dir, 'SETTINGS.json')).read())
    train_path = os.path.join(project_dir, settings['TRAIN_DATA_PATH'])
    train_data = pd.read_csv(train_path)

    train_imgs = train_data.drop('label', axis=1)
    one_hot_target = pd.get_dummies(train_data['label'], prefix='dig')
    train_data = pd.concat([train_imgs, one_hot_target], axis=1, join='inner')

    # Run training
    if model == 'logreg':
        logreg.run(train_data, None, True, False)
    else:
        convnn.run(train_data, None, True, False)


if __name__ == '__main__':
    main()
