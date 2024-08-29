import os

# PACKAGE
# Root directory of package.
PACKAGE = os.path.dirname(__file__)

# PART LABEL CSV
# CSV file contains labels corresponding to LDraw parts
# LDraw parts are identified by a dat filename
PART_LABEL_CSV = os.path.join(PACKAGE, "LDraw/part/part_label.csv")

# PART DETAILS CSV
# CSV file contains essential informations about LDraw parts
# LDraw parts are identified by a dat filename
PART_DETAILS_CSV = os.path.join(PACKAGE, "LDraw/part/part_details.csv")

# PART .DAT DIRECTORY
# Directory contains all part .dat files.
PART_DAT_DIR = os.path.join(PACKAGE, "LDraw/part/dat")

# PART GEOMETRY DIRECTORY
# Directory contains all part geometry files. File could be for
# example .stl file.
PART_GEOM_DIR = os.path.join(PACKAGE, "LDraw/part/geom")

# BRICK MODELS DIRECTORY
# Directory of brick classification models saved as .pth files.
BRICK_MODELS_DIR = os.path.join(PACKAGE, "models/brick_classification")

# BRICK MODULES FILE
# Python module defining and operating on brick classification modules.
BRICK_MODULES_FILE = os.path.join(
    PACKAGE, "models/brick_classification_models.py")

# DATA DIRECTORY
# Top level directory containng all data for training and testing
# neural networks.
DATA_DIR = os.path.join(PACKAGE, "data")

# BRICK CLASSFICATION DATA DIRECTORY
# Directry containing data for training and testing brick
# classification networks.
BRICK_CLASSFICATION_DATA_DIR = os.path.join(DATA_DIR, "brick_classification")

# SAMPLE MODELS DIRECTORY
# Directory containing sample geometry objects for testing.
SAMPLE_MODELS_DIR = os.path.join(PACKAGE, "sample_models")
