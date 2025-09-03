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
PADDING = np.array([0, 0, 0])

# LDRAW_BRICK_OFFSET
# Offset possition for every LDraw part. For example brick 3005 bottom height 
# is 0, but brick 54200 bottom height is -24. Offset fix these differences.
LDRAW_PART_OFFSET = {"3005": np.array([0, 0, 0]), 
                      "54200": np.array([0, 24, 0]),}