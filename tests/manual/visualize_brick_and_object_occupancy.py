import pyvista as pv
import numpy as np
from nelegolizer.data import LDrawFile, initilize_parts, BrickOccupancy, ObjectOccupancy
from nelegolizer import const

import nelegolizer.utils.grid as grid
import nelegolizer.utils.voxelization as uvox

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
lbm = ldf.models[0]
bricks = lbm.as_bricks()

# bo and oo with the same voxel grid
bo = BrickOccupancy.from_bricks(bricks)
oo = ObjectOccupancy(bo.voxel_grid)

plotter = pv.Plotter(shape=(1, 2))

# bo plot
bo_bricks_grid = uvox.from_grid(bo.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
bo_voxels_grid = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.subplot(0, 0)
plotter.add_title("brick occupancy", 8)
plotter.add_mesh(bo_bricks_grid, show_edges=True, color="white", opacity=0.5)
voxels1_mesh = bo_voxels_grid
voxels1_mesh = voxels1_mesh.translate(np.array([-0.32, -0.32, -0.32]), inplace=False)
plotter.add_mesh(voxels1_mesh, show_edges=True, color="blue")

# oo plot
oo_bricks_grid = uvox.from_grid(oo.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
oo_voxel_grid = uvox.from_grid(oo.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.subplot(0, 1)
plotter.add_title("object occupancy", 8)
plotter.add_mesh(oo_bricks_grid, show_edges=True, color="white", opacity=0.5)
voxels_mesh = oo_voxel_grid
voxels_mesh = voxels_mesh.translate(np.array([-0.32, -0.32, -0.32]), inplace=False)
plotter.add_mesh(voxels_mesh, show_edges=True, color="blue")

plotter.show()