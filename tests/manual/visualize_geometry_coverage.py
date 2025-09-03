from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickCoverage
from nelegolizer.data._GeometryCoverage import GeometryCoverage

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
ldm = ldf.models[0]
bricks = ldm.as_bricks()

#brick_occupany = BrickOccupancy.from_bricks(bricks)
bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, side_extension=1)

geometry = GeometryCoverage(bc.voxel_grid, bottom_extension=3, side_extension=1)
#go = GeometryCoverage(bc.voxel_grid)


print(f"geometry ext voxel shape {geometry.ext_voxel_grid.shape}")
geometry.ext_voxel_grid[14, 0, 19] = True
geometry.ext_voxel_grid[14, 1, 19] = True
geometry.ext_voxel_grid[14, -1, 19] = True
geometry.ext_voxel_grid[14, -2, 19] = True

geometry.voxel_grid[14, 0, 19] = True
geometry.voxel_grid[14, -1, 19] = True

#bc.voxel_grid[14, 0, 19] = True
#bc.voxel_grid[14, 1, 19] = True
#bc.voxel_grid[14, -1, 19] = True
#bc.voxel_grid[14, -2, 19] = True

geometry.brick_grid[3, -1, 3] = True
geometry.brick_grid[3, -2, 3] = True
geometry.brick_grid[3, -3, 3] = True
geometry.brick_grid[3, 0:3, 3] = True

mesh = pv.MultiBlock([brick.mesh for brick in bricks]).combine()

bricks = uvox.from_grid(geometry.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
voxels = uvox.from_grid(geometry.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
#voxels = uvox.from_grid(bc.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter = pv.Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_title("mesh", 8)
plotter.add_mesh(mesh)

plotter.subplot(0, 1)
plotter.add_title("brick occupancy", 8)
plotter.add_mesh(bricks, show_edges=True, color="white")

plotter.subplot(0, 2)
plotter.add_title("brick voxel occupancy", 8)
plotter.add_mesh(voxels, show_edges=True, color="white")


#plotter.add_mesh(voxels, show_edges=True, color="white")
plotter.show()