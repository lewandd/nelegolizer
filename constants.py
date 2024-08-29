import numpy as np

# BRICK_UNIT_MESH_SHAPE
# Shape of single 1x1x1 brick mesh.
# Standard is 0.8cm x 0.8cm base and 0.96cm height. (It is assumed that
# mesh models maintain proper scale)
BRICK_UNIT_MESH_SHAPE = np.array([0.8, 0.96, 0.8])

# BRICK_UNIT_RESOLUTION
# Resolution of cuboid representing a 1x1x1 brick made of voxels.
BRICK_UNIT_RESOLUTION = np.array([4, 4, 4])

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

# TOP_LEVEL_BRICK_SHAPE
# Shape of the smallest subspace that can include any brick.
# Space is defined by a cuboid made of 1x1x1 bricks.
TOP_LEVEL_BRICK_SHAPE = np.array([1, 1, 1])

# TOP_LEVEL_BRICK_RESOLUTION
# Shape of the smallest subspace that can include any brick.
# Space is defined by a cuboid made of voxels.
TOP_LEVEL_BRICK_RESOLUTION = TOP_LEVEL_BRICK_SHAPE * BRICK_UNIT_RESOLUTION

# PADDING
# Extended space around 3-dim voxel grid. Subspaces with padding are inputs
# for neural networks.
PADDING = np.array([1, 1, 1])
