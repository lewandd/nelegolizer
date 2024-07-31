import os
import numpy as np

# path to library main folder
PATH = os.path.dirname(os.path.dirname(__file__))
DIR_LABELS = '/data/generated/'

BRICK_UNIT_SHAPE = np.array([0.8, 0.96, 0.8])
BRICK_UNIT_RESOLUTION = np.array([4, 4, 4])
VOXEL_UNIT_SHAPE = BRICK_UNIT_SHAPE / BRICK_UNIT_RESOLUTION
BRICK_SHAPES = [np.array([1, 1, 1])]
BRICK_SHAPE_BOUND = np.array([1, 1, 1])