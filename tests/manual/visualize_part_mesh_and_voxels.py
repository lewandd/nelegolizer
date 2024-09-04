import pyvista as pv

from nelegolizer.data import part_by_filename
from nelegolizer.utils import voxelization
from nelegolizer import const

part = part_by_filename["54200.dat"]
plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(part.mesh)
plotter.add_title("54200.stl")

plotter.subplot(0, 1)
voxels = voxelization.from_grid(part.grid,
                                voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter.add_mesh(voxels, show_edges=True)
plotter.add_title("Voxelized 54200.stl")

plotter.show()
