"""
Visualize mesh voxelization with different voxel_mesh_shape parameter.

Plot voxelized mesh with voxel shape: [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]
and const.VOXEL_MESH_SHAPE.
"""
import numpy as np
import pyvista as pv

from nelegolizer.utils import voxelization
from nelegolizer import const

# load mesh
reader = pv.get_reader("fixtures/cone.obj")
mesh = reader.read()
mesh = mesh.scale(1)
plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 800))

# voxel_mesh_shape=[0.2, 0.2, 0.2]
pv_voxels = voxelization.from_mesh(mesh,
                                   voxel_mesh_shape=np.array([0.2, 0.2, 0.2]))
plotter.subplot(0, 0)
plotter.add_mesh(pv_voxels, show_edges=True)
plotter.show_grid()
plotter.add_title("voxel_mesh_shape=[0.2, 0.2, 0.2]", 6)

# voxel_mesh_shape=[0.5, 0.5, 0.5]
pv_voxels2 = voxelization.from_mesh(mesh,
                                    voxel_mesh_shape=np.array([0.5, 0.5, 0.5]))
plotter.subplot(0, 1)
plotter.add_mesh(pv_voxels2, show_edges=True)
plotter.show_grid()
plotter.add_title("voxel_mesh_shape=[0.5, 0.5, 0.5]", 6)

# voxel_mesh_shape=const.VOXEL_MESH_SHAPE
pv_voxels3 = voxelization.from_mesh(mesh,
                                    voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.subplot(0, 2)
plotter.add_mesh(pv_voxels3, show_edges=True)
plotter.show_grid()
plotter.add_title("voxel_mesh_shape=const.VOXEL_MESH_SHAPE", 6)

plotter.show()
