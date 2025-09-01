from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickOccupancy
from nelegolizer.data import BrickCoverage
from nelegolizer.data._GeometryCoverage import GeometryCoverage

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
ldm = ldf.models[0]
bricks = ldm.as_bricks()

#brick_occupany = BrickOccupancy.from_bricks(bricks)
bc = BrickCoverage.from_bricks(bricks)

go = GeometryCoverage(bc.voxel_grid)
#go = GeometryCoverage(bc.voxel_grid)


print(f"geometry ext voxel shape {go.ext_voxel_grid.shape}")
go.ext_voxel_grid[14, 0, 19] = True
go.ext_voxel_grid[14, 1, 19] = True
go.ext_voxel_grid[14, -1, 19] = True
go.ext_voxel_grid[14, -2, 19] = True

go.voxel_grid[14, 0, 19] = True
go.voxel_grid[14, -1, 19] = True

bc.voxel_grid[14, 0, 19] = True
bc.voxel_grid[14, 1, 19] = True
bc.voxel_grid[14, -1, 19] = True
bc.voxel_grid[14, -2, 19] = True

mesh = pv.MultiBlock([brick.mesh for brick in bricks]).combine()
bricks = uvox.from_grid(go.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)

voxels = uvox.from_grid(go.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
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