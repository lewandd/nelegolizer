from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickCoverage

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
ldm = ldf.models[0]
bricks = ldm.as_bricks()

bo = BrickCoverage.from_bricks(bricks, bottom_extension=3, side_extension=1)

print(f"shape {bo.ext_voxel_grid.shape}")
pos_top = (14,0,19)
pos_top2 = (14,1,19)
pos_top3 = (14,2,19)
pos_top4 = (14,3,19)
pos_bot = (14,-1,19)
pos_bot2 = (14,-2,19)
pos_bot3 = (14,-3,19)
bo.ext_voxel_grid[pos_top] = True
bo.ext_voxel_grid[pos_top2] = True
bo.ext_voxel_grid[pos_top3] = True
bo.ext_voxel_grid[pos_top4] = True
bo.ext_voxel_grid[pos_bot] = True
bo.ext_voxel_grid[pos_bot2] = True
bo.ext_voxel_grid[pos_bot3] = True

bo.brick_grid[3, -1, 3] = True
bo.brick_grid[3, -2, 3] = True
bo.brick_grid[3, -3, 3] = True
bo.brick_grid[3, 0:3, 3] = True

mesh = pv.MultiBlock([brick.mesh for brick in bricks]).combine()
bricks = uvox.from_grid(bo.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
voxels = uvox.from_grid(bo.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
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