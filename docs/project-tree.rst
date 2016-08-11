Project Structure
====================

The project structure can be viewed bellow with some description:

.. code-block:: none

	├── LICENSE
	├── Makefile           <- Makefile with commands like `make data` or `make train`
	├── README.md          <- The top-level README for developers using this project.
	├── SETTINGS.json      <- This file specifies the path to the train, test, model, and output
	│                          directories. 
	├── data
	│   ├── interim        <- Intermediate data that has been transformed.
	│   ├── processed      <- The final, canonical data sets for modeling.
	│   └── raw            <- The original, immutable data dump.
	│
	├── docs               <- A default Sphinx project; see sphinx-doc.org for details
	│
	├── models             <- Trained and serialized models, model predictions, or model summaries
	│
	├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
	│                         the creator's initials, and a short `-` delimited description, e.g.
	│                         `1.0-jqp-initial-data-exploration`.
	│
	├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
	│                         generated with `pip freeze > requirements.txt`
	│
	├── src                <- Source code for use in this project.
	│   ├── __init__.py    <- Makes src a Python module
	│   │
	│   ├── data           <- Scripts to download or generate data
	│   │   └── make_dataset.py
	│   │
	│   ├── features       <- Scripts to turn raw data into features for modeling
	│   │   └── build_features.py
	│   │
	│   └── models         <- Scripts to train models and then use trained models to make
	│       │                 predictions
	│       ├── predict.py
	│       └── train.py
	│
	├── submissions        <- Kaggle submissions
	│   
	└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org