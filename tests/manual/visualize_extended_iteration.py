import pyvista as pv
import numpy as np
from nelegolizer.data import LDrawFile, initilize_parts, BrickCoverage
from nelegolizer.utils.brick import compute_bounds
from nelegolizer.data._GeometryCoverage import GeometryCoverage
from nelegolizer import const
import nelegolizer.utils.voxelization as uvox
from nelegolizer.utils.conversion import *
from nelegolizer.legolizer.iterator import find_next_to_cover_net, place_brick
from nelegolizer.model.dataset_generation import make_brick_variants, get_label

initilize_parts()

#filename = "fixtures/impossible_trophy.mpd"
#filename = "fixtures/church.mpd"
filename = "fixtures/simple_mountain.mpd"

ldf = LDrawFile.load(filename)
lbm = ldf.models[0]
bricks = lbm.as_bricks()

mins, maxs = compute_bounds(bricks)

for brick in bricks:
    brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([2, 4, 2])).astype(int))
    print(f"{brick.id} position {brick.position}")

filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]
gc = GeometryCoverage(interior_voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
training_bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

analyzed = {1: np.zeros_like(training_bc.brick_grid),
            3: np.zeros_like(training_bc.brick_grid)}
ntc1, ntc3 = find_next_to_cover_net(gc, training_bc, analyzed)
num_found = 0
num_all = 0

while ntc1 is not None or ntc3 is not None:
    if ntc3 is not None and (ntc1 is None or ntc3[1] >= ntc1[1]):
        ntc, network_type = ntc3, 3
    else:
        ntc, network_type = ntc1, 1
    x, y, z = ntc

    print(f"Using network type{network_type}...")

    if network_type == 1:
        shape = np.array([3, 3, 3])
        shape_top_ext = 0
        placement_pos = np.array([x, y-1, z])
    else:
        shape = np.array([5, 5, 5])
        shape_top_ext = 2
        placement_pos = np.array([x, y-3, z])
        
    looking_pos = np.array([x, y-1, z])
    shape_side_ext = int(np.round((shape[0]-1)/2))

    print(f"Looked at {looking_pos} location for a brick")

    placed = False
    if any(training_bc.is_placement_available(b) 
        for b in make_brick_variants(placement_pos, network_type).values()):
        label = get_label(filled_bc, looking_pos, placement_pos, network_type, bricks)
        
        if label != 0:
            placed = place_brick(label, placement_pos, network_type, training_bc)

    num_all += 1
    analyzed[network_type][x, y, z] = True
    ntc1, ntc3 = find_next_to_cover_net(gc, training_bc, analyzed)

    # 30 15 30 i 18 9 18

    ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
    ext_vu_shape = ext_bu_to_vu(shape)

    channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                 ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                 ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    channel2 = training_bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                          ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                          ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    print(f"Rozmiary channeli {channel1.shape}, {channel2.shape}")
    plotter = pv.Plotter(shape=(1, 2))
    # subplot - whole voxels scene with moving kernel
    plotter.subplot(0, 0)

    if network_type == 1:
        shape_of_selection = np.array([5, 3, 5]) * const.VOXEL_MESH_SHAPE
    if network_type == 3:
        shape_of_selection = np.array([5, 9, 5]) * const.VOXEL_MESH_SHAPE
        
    selection_pos = (np.array([6, 3, 6])*np.array([x, y-shape[1]+3, z])-np.array([0,1,0]))*const.VOXEL_MESH_SHAPE
    selection_mesh = uvox.from_grid(np.array([[[1]]]), voxel_mesh_shape=shape_of_selection)
    selection_mesh.translate(selection_pos, inplace=True)
    if network_type == 1:
        plotter.add_mesh(selection_mesh, show_edges=True, color="orange", opacity=0.2)
    else:
        plotter.add_mesh(selection_mesh, show_edges=True, color="purple", opacity=0.2)

    #voxels_scene_mesh = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxels_mesh = uvox.from_grid(training_bc.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxels_mesh.translate(np.array([0, -0.16, 0]), inplace=True)
    plotter.add_mesh(voxels_mesh, show_edges=True, color="white")



    plotter.subplot(0, 1)
    try:
        channel1_mesh = uvox.from_grid(channel1, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
        plotter.add_mesh(channel1_mesh, show_edges=True, color="blue", opacity=0.4)
        channel2_mesh = uvox.from_grid(channel2, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
        plotter.add_mesh(channel2_mesh, show_edges=True, color="white", opacity=1)
    except Exception:
        pass
    
    cpos = [(4, -11.5, 4), 
            (1.28, 0.64, 1.28), 
            (-1, -0.17, 0.16)]
    cpos2 = [(-3.15, -14.13, -4.28), 
            (2.00, 1.36, 2.4), 
            (0.37, -0.47, 0.8)]
    cpos_moving = [(4+x*const.BRICK_UNIT_MESH_SHAPE[0], -11.5+y*const.BRICK_UNIT_MESH_SHAPE[1], 4+z*const.BRICK_UNIT_MESH_SHAPE[2]), 
            (1.28+x*const.BRICK_UNIT_MESH_SHAPE[0], 0.64+y*const.BRICK_UNIT_MESH_SHAPE[1], 1.28+z*const.BRICK_UNIT_MESH_SHAPE[2]), 
            (-1, -0.17, 0.16)]
    plotter.subplot(0, 0)
    plotter.camera_position = cpos2#_moving
    plotter.subplot(0, 1)
    plotter.camera_position = cpos2
    plotter.show(window_size=(1800, 800))

print(f"Found {num_found} bricks blocks. All locations looked: {num_all}.")
