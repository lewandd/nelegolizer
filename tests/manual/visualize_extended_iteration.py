import pyvista as pv
import numpy as np
from nelegolizer.data import LDrawFile, initilize_parts, BrickCoverage
from nelegolizer.utils.brick import compute_bounds
from nelegolizer.data._GeometryCoverage import GeometryCoverage
import nelegolizer.utils.voxelization as uvox
from nelegolizer.utils.conversion import bu_to_mesh, ext_bu_to_vu
from nelegolizer.constants import VU, BU
from nelegolizer.legolizer.iterator import find_next_pos_to_cover, place_brick, make_brick_variants
from nelegolizer.model.dataset_generation import get_label
import yaml

initilize_parts()

#filename = "fixtures/impossible_trophy.mpd"
filename = "fixtures/church.mpd"
#filename = "fixtures/simple_mountain.mpd"

ldf = LDrawFile.load(filename)
lbm = ldf.models[0]
bricks = lbm.as_bricks()

mins, maxs = compute_bounds(bricks)

for brick in bricks:
    brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([2, 4, 2])).astype(int))
    print(f"{brick.id} position {brick.position}")

# load config
with open("../../configs/datasets/smountain1.dataset.yaml") as f:
    config = yaml.safe_load(f)

filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]
gc = GeometryCoverage(interior_voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
training_bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

analyzed = {subset: np.zeros_like(training_bc.brick_grid) for subset in config['dataset']['subsets']}
ntcs = find_next_pos_to_cover(gc, training_bc, analyzed, config)
num_found = 0
num_all = 0

while any(x is not None for x in ntcs.values()):
    not_none_ntcs = dict((k, v) for k, v in ntcs.items() if v is not None)
    subset_used = list(not_none_ntcs.keys())[0]
    pos = not_none_ntcs[subset_used]
    for subs, indices in not_none_ntcs.items():
        if pos[1] < indices[1]:
            subset_used = subs
            pos = indices
    x, y, z = pos

    #print(f"Using network type{network_type}...")


    shape = np.array(config['dataset']['subsets'][subset_used]['iteration']['group_shape'])
    #print("proposed shape", prop_shape)
    shape_top_ext = config['dataset']['subsets'][subset_used]['iteration']['shape_top_ext']
    placement_pos = np.array([x, y-1-shape_top_ext, z])
        
    looking_pos = np.array([x, y-1, z])
    shape_side_ext = int(np.round((shape[0]-1)/2))

    #print(f"Looked at {looking_pos} location for a brick")

    placed = False
    brick_variants = make_brick_variants(placement_pos, subset_used, config)
    
    if any(training_bc.is_placement_available(b) for b in brick_variants):
        label = get_label(filled_bc, looking_pos, placement_pos, config, subset_used, bricks)
        if label != 0:
            #placed = place_brick(label, placement_pos, network_type, training_bc)
            placed = place_brick(label, placement_pos, config, subset_used, training_bc)

    num_all += 1
    analyzed[subset_used][x, y, z] = True
    ntcs = find_next_pos_to_cover(gc, training_bc, analyzed, config)

    # 30 15 30 i 18 9 18

    ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
    ext_vu_shape = ext_bu_to_vu(shape)

    channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                 ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                 ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    channel2 = training_bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                          ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                          ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    #print(f"Rozmiary channeli {channel1.shape}, {channel2.shape}")
    plotter = pv.Plotter(shape=(1, 2))
    # subplot - whole voxels scene with moving kernel
    plotter.subplot(0, 0)

    if subset_used == 'subset1':
        shape_of_selection = np.array([5, 9, 5]) * VU
    if subset_used == 'subset2':
        shape_of_selection = np.array([5, 3, 5]) * VU

    selection_pos = (np.array([6, 3, 6])*np.array([x, y-shape[1]+3, z])-np.array([0,1,0]))*VU
    selection_mesh = uvox.from_grid(np.array([[[1]]]), voxel_mesh_shape=shape_of_selection)
    selection_mesh.translate(selection_pos, inplace=True)
    if subset_used == 'subset2':
        plotter.add_mesh(selection_mesh, show_edges=True, color="orange", opacity=0.2)
    else:
        plotter.add_mesh(selection_mesh, show_edges=True, color="purple", opacity=0.2)

    #voxels_scene_mesh = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=VU)
    voxels_mesh = uvox.from_grid(training_bc.ext_voxel_grid, voxel_mesh_shape=VU)
    voxels_mesh.translate(np.array([0, -0.16, 0]), inplace=True)
    plotter.add_mesh(voxels_mesh, show_edges=True, color="white")



    plotter.subplot(0, 1)
    try:
        channel1_mesh = uvox.from_grid(channel1, voxel_mesh_shape=VU)
        plotter.add_mesh(channel1_mesh, show_edges=True, color="blue", opacity=0.4)
        channel2_mesh = uvox.from_grid(channel2, voxel_mesh_shape=VU)
        plotter.add_mesh(channel2_mesh, show_edges=True, color="white", opacity=1)
    except Exception:
        pass
    
    cpos = [(4, -11.5, 4), 
            (1.28, 0.64, 1.28), 
            (-1, -0.17, 0.16)]
    cpos2 = [(-3.15, -14.13, -4.28), 
            (2.00, 1.36, 2.4), 
            (0.37, -0.47, 0.8)]
    cpos_moving = [(4+x*BU[0], -11.5+y*BU[1], 4+z*BU[2]), 
            (1.28+x*BU[0], 0.64+y*BU[1], 1.28+z*BU[2]), 
            (-1, -0.17, 0.16)]
    plotter.subplot(0, 0)
    plotter.camera_position = cpos2#_moving
    plotter.subplot(0, 1)
    plotter.camera_position = cpos2
    plotter.show(window_size=(1800, 800))

print(f"Found {num_found} bricks blocks. All locations looked: {num_all}.")
