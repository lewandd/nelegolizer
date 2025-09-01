import numpy as np

# BRICK_UNIT_MESH_SHAPE
# Shape of single 1x1x1 brick mesh.
# Standard is 0.8cm x 0.8cm base and 0.96cm height. (It is assumed that
# mesh models maintain proper scale)
BRICK_UNIT_MESH_SHAPE = np.array([0.8, 0.32, 0.8])

# BRICK_UNIT_LDU_SHAPE
# Shape of single 1x1x1 brick with LDraw Unit.
LDU = np.array([0.04, 0.04, 0.04])
BRICK_UNIT_LDU_SHAPE = BRICK_UNIT_MESH_SHAPE / LDU

# BRICK_UNIT_RESOLUTION
# Resolution of cuboid representing a 1x1x1 brick made of voxels.
BRICK_UNIT_RESOLUTION = np.array([5, 2, 5])

# BRICK_UNIT_VOLUME
# Quantity of voxels representing a 1x1x1 brick.
BRICK_UNIT_VOLUME = (BRICK_UNIT_RESOLUTION[0] *
                     BRICK_UNIT_RESOLUTION[1] *
                     BRICK_UNIT_RESOLUTION[2])

# VOXEL_MESH_SHAPE
# Shape of single voxel mesh.
# Calculated by getting a fraction of 1x1x1 brick mesh shape.
VOXEL_MESH_SHAPE = BRICK_UNIT_MESH_SHAPE / BRICK_UNIT_RESOLUTION

# BRICK_SHAPES
# List if possible brick shapes.
# Each shape is defined by a cuboid made of 1x1x1 bricks.
BRICK_SHAPES = [np.array([1, 1, 1])]

# LCH
# Least Common Hull of all possible used bricks layouts
LCH = np.array([1, 1, 1])

# PADDING
# Extended space around 3-dim voxel grid. Subspaces with padding are inputs
# for neural networks.
PADDING = np.array([2, 2, 2])

# LDRAW_BRICK_OFFSET
# Offset possition for every LDraw part. For example brick 3005 bottom height 
# is 0, but brick 54200 bottom height is -24. Offset fix these differences.
LDRAW_PART_OFFSET = {"3005": np.array([0, 0, 0]), 
                      "54200": np.array([0, 24, 0]),}


# -----------------------------------------------------------------------------

import os

# PACKAGE
# Root directory of package.
PACKAGE = os.path.dirname(__file__)

# PART DATA CSV
# CSV file contains essential informations about LDraw parts
# LDraw parts are identified by a size with label or by a dat filename.
PART_DATA_CSV = os.path.join(PACKAGE, "LDraw/part/part_data.csv")

# PARTS CSV
# CSV file containing essential informations about LDraw parts
# LDraw parts are identified by a part id
PARTS_CSV = os.path.join(PACKAGE, "LDraw/part/parts.csv")

PARTS2_CSV = os.path.join(PACKAGE, "LDraw/part/parts2.csv")

# DIVISIONS CSV
# CSV file containing essential informations about possible divisions
# Divisions are identified by a division id
DIVISIONS_CSV = os.path.join(PACKAGE, "models/divisions.csv")

SUBSHAPES_CSV = os.path.join(PACKAGE, "models/subshapes.csv")

# LABELS CSV
# CSV file containing "type" (brick or division) 
# and "type_id" (brick or division id) coresponding to network with "shape" 
# and result "labels"
LABELS_CSV = os.path.join(PACKAGE, "models/labels.csv")

LABELS2_CSV = os.path.join(PACKAGE, "models/labels2.csv")

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