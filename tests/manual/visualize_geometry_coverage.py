from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickCoverage
from nelegolizer.data._GeometryCoverage import GeometryCoverage
from nelegolizer.data._BrickCoverage import compute_bounds
from nelegolizer.utils.grid import get_fill, bu_to_mesh
import numpy as np

initilize_parts()

#filename = "fixtures/church.mpd"
filename = "fixtures/impossible_trophy.mpd"

ldf = LDrawFile.load(filename)
ldm = ldf.models[0]
bricks = ldm.as_bricks()

mins, maxs = compute_bounds(bricks)
for brick in bricks:
    #print()
    #print(f"old position {brick.position}")
    #print(f"({brick.position} - {mins}).astype(int) = {(brick.position - mins).astype(int)}")
    brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([1, 4, 1])).astype(int))
    #print(f"new position {brick.position}")

#brick_occupany = BrickOccupancy.from_bricks(bricks)
bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, side_extension=1, top_extension=4)
interior_voxel_grid = bc.voxel_grid[5:-5, 8:-6, 5:-5]
geometry = GeometryCoverage(interior_voxel_grid, bottom_extension=3, side_extension=1, top_extension=4)
#go = GeometryCoverage(bc.voxel_grid)


print(f"bc ext voxel shape {bc.ext_voxel_grid.shape}")
print(f"geometry ext voxel shape {geometry.ext_voxel_grid.shape}")
#geometry.ext_voxel_grid[14, 0, 19] = True
#geometry.ext_voxel_grid[14, 1, 19] = True
#geometry.ext_voxel_grid[14, -1, 19] = True
#geometry.ext_voxel_grid[14, -2, 19] = True

#geometry.voxel_grid[14, 0, 19] = True
#geometry.voxel_grid[14, -1, 19] = True

#bc.voxel_grid[14, 0, 19] = True
#bc.voxel_grid[14, 1, 19] = True
#bc.voxel_grid[14, -1, 19] = True
#bc.voxel_grid[14, -2, 19] = True

#geometry.brick_grid[3, -1, 3] = True
#geometry.brick_grid[3, -2, 3] = True
#geometry.brick_grid[3, -3, 3] = True
#geometry.brick_grid[3, 0:3, 3] = True

mesh = pv.MultiBlock([brick.mesh for brick in bricks]).combine()

bricks = uvox.from_grid(geometry.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
#voxels = uvox.from_grid(bc.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
voxels = uvox.from_grid(geometry.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
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