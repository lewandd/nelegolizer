"""
Test integrity of grid.from_pv_voxels and voxelization.from_grid functions. Combination of them both should return input voxels.

Plot voxels from basic voxelization and voxels after grid.from_pv_voxels and voxelization.from_grid functions.
"""
import pyvista as pv

from nelegolizer.utils import voxelization, grid
from nelegolizer import const

# load model
reader = pv.get_reader("fixtures/cone.obj")
mesh = reader.read()

# plot voxels
plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
pv_voxels = voxelization.from_mesh(mesh, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.add_mesh(pv_voxels, show_edges=True)
plotter.show_grid()
plotter.add_title("Standard voxelize from mesh", 8)

# voxels into grid, voxelize from grid and plot
plotter.subplot(0, 1)
voxel_grid = grid.from_pv_voxels(pv_voxels)
pv_voxels2 = voxelization.from_grid(voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.add_mesh(pv_voxels2, show_edges=True)
plotter.show_grid()
plotter.add_title("Voxels into grid and then voxelize this grid", 8)

plotter.show()
