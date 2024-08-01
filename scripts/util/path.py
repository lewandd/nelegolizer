import os

PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# models
BRICK_MODELS_DIR = os.path.join(PACKAGE, "models/brick_classification")

# data
DATA_DIR = os.path.join(PACKAGE, "data")
BRICK_CLASSFICATION_DATA_DIR = os.path.join(DATA_DIR, "brick_classification")