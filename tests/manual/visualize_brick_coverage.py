from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickCoverage
import numpy as np

initilize_parts()

filename = "fixtures/impossible_trophy.mpd"
#filename = "fixtures/church.mpd"

ldf = LDrawFile.load(filename)
ldm = ldf.models[0]
bricks = ldm.as_bricks()

bo_old = BrickCoverage.from_bricks(bricks, bottom_extension=3, side_extension=1, top_extension=1)
bo = BrickCoverage(bo_old.shape-np.array([2, 4, 2]), bottom_extension=3, side_extension=1, top_extension=1)
bo.pos_min = bo_old.pos_min
bo.pos_max = bo_old.pos_max

bo.place_brick(bricks[0])
bo.place_brick(bricks[1])
bo.place_brick(bricks[2])
bo.place_brick(bricks[3])
bo.place_brick(bricks[4])
bo.place_brick(bricks[5])
bo.place_brick(bricks[6])
bo.place_brick(bricks[7])
bo.place_brick(bricks[8])
bo.place_brick(bricks[9])
bo.place_brick(bricks[10])
bo.place_brick(bricks[11])
bo.place_brick(bricks[12])
bo.place_brick(bricks[13])
bo.place_brick(bricks[14])
bo.place_brick(bricks[15])
bo.place_brick(bricks[16])
bo.place_brick(bricks[17])
bo.place_brick(bricks[18])
bo.place_brick(bricks[19])
bo.place_brick(bricks[20])
bo.place_brick(bricks[21])
bo.place_brick(bricks[22])
bo.place_brick(bricks[23])
bo.place_brick(bricks[24])
bo.place_brick(bricks[25])
bo.place_brick(bricks[26])
#bo.ext_voxel_grid[6, :2, 6] = True
#bo.ext_voxel_grid[6, -2:, 6] = True

#print(bricks[7].mesh_position)
#print(bricks[8].mesh_position)

#bo.brick_grid[3, -2:, 3] = True
#bo.brick_grid[3, 0:2, 3] = True

mesh = pv.MultiBlock([brick.mesh for brick in bricks]).combine()
voxels = uvox.from_grid(bo.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
plotter = pv.Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_title("mesh", 8)
plotter.add_mesh(mesh)

plotter.subplot(0, 1)
plotter.add_title("BU coverage", 8)

bricks = uvox.from_grid(bo.brick_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)

try:
    bottom_studs = uvox.from_grid(bo.bottom_available_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
    plotter.add_mesh(bottom_studs, show_edges=True, color="orange", opacity=0.5)
except TypeError:
    pass

try:
    bottom3_studs = uvox.from_grid(bo.bottom3_available_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
    plotter.add_mesh(bottom3_studs, show_edges=True, color="yellow", opacity=0.5)
except TypeError:
    pass

try:
    top_studs = uvox.from_grid(bo.top_available_grid, voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
    plotter.add_mesh(top_studs, show_edges=True, color="green", opacity=0.5)
except TypeError:
    pass

plotter.add_mesh(bricks, show_edges=True, color="white")



plotter.subplot(0, 2)
plotter.add_title("extended voxel grid", 8)
plotter.add_mesh(voxels, show_edges=True, color="white")


#plotter.add_mesh(voxels, show_edges=True, color="white")
plotter.show()