"""
Visualize basic voxelization.

Plot original mesh and voxelized mesh.
"""
import pyvista as pv
from nelegolizer.utils import voxelization
import numpy as np

# load model
reader = pv.get_reader("fixtures/cone.obj")
mesh = reader.read()

# plot mesh
plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(mesh)
plotter.show_grid()
plotter.add_title("Mesh", 12)

# voxelize with unit_shape=(1, 1, 1) and plot
pv_voxels = voxelization.from_mesh(mesh,
                                   voxel_mesh_shape=np.array([0.1, 0.1, 0.1]))
plotter.subplot(0, 1)
plotter.add_mesh(pv_voxels, opacity=1)
plotter.show_grid()
plotter.add_title("Voxelized mesh", 12)
plotter.show()
