import click
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

import logreg
import convnn


@click.command()
@click.option('--model', type=click.Choice(['logreg', 'convnn']), default='convnn')
def main(model):
    # Load data
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(
        open(os.path.join(project_dir, 'SETTINGS.json')).read()
    )

    test_path = os.path.join(project_dir, settings['TEST_DATA_PATH'])
    test_data = pd.read_csv(test_path)

    # Run training
    if model == 'logreg':
        logreg.run(None, test_data, False, True)
    else:
        convnn.run(None, test_data, False, True)


if __name__ == '__main__':
    main()