import os

PACKAGE = os.path.dirname(__file__)

# Part
PART_LABEL_CSV = os.path.join(PACKAGE, "LDraw/part/part_label.csv")
PART_DETAILS_CSV = os.path.join(PACKAGE, "LDraw/part/part_details.csv")
PART_DAT_DIR = os.path.join(PACKAGE, "LDraw/part/dat")
PART_GEOM_DIR = os.path.join(PACKAGE, "LDraw/part/geom")

# Brick Models
BRICK_MODELS_DIR = os.path.join(PACKAGE, "models/brick_classification")
BRICK_MODULES_FILE = os.path.join(PACKAGE, "models/brick_classification_models.py")

# Data
DATA_DIR = os.path.join(PACKAGE, "data")
BRICK_CLASSFICATION_DATA_DIR = os.path.join(DATA_DIR, "brick_classification")

# Additional
SAMPLE_MODELS_DIR = os.path.join(PACKAGE, "sample_models")