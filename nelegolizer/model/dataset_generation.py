from ..data import LDrawFile, LegoBrick, GeometryCoverage, LDrawFile, BrickCoverage, LegoBrick, part_by_id
from ..utils import brick as utils_brick
from ..legolizer.iterator import find_next_pos_to_cover, place_brick, make_brick_variants
from ..utils.conversion import bu_to_mesh, ext_bu_to_vu
import numpy as np
import json
import copy

def sample_to_str(channel1: np.ndarray, channel2: np.ndarray, brick_id: str, rotation: int) -> str:
    """
    Convert two 3D bool ndarrays and single int label into one-line str (JSON)
    """
    assert channel1.shape == channel2.shape, "Obie siatki muszą mieć ten sam wymiar"
    assert channel1.dtype == bool and channel2.dtype == bool, "Siatki powinny być typu bool"

    data_dict = {
        "channel1": channel1.astype(int).flatten().tolist(),
        "channel2": channel2.astype(int).flatten().tolist(),
        "shape": channel1.shape,
        "brick_id": brick_id,
        "rotation": rotation
    }
    return json.dumps(data_dict)

def save_dataset(samples: list, filename: str):
    """
    Save samples to txt file.
    """
    with open(filename, "w") as f:
        for s in samples:
            f.write(s + "\n")

#def get_label(filled_bc, looking_pos, placement_pos, config, subset, bricks):
#    brick = filled_bc.get_brick_at(bricks, looking_pos)
#    labels = config['dataset']['subsets'][subset]['labels']
#    if brick is not None and brick.id in labels.keys():
#        proper_pos = filled_bc.get_brick_position(brick)
#        rot = brick.rotation
#        if proper_pos[0] == placement_pos[0] - 1:
#            rot = 180
#        if proper_pos[2] == placement_pos[2] - 1:
#            rot = 270
#        return labels[brick.id][rot]
#    else:
#        return 0
    
def get_brick_id_rotation(filled_bc, looking_pos, placement_pos, config, subset, bricks):
    brick = filled_bc.get_brick_at(bricks, looking_pos)
    bricks_pool = config['dataset']['subsets'][subset]['bricks']
    if brick is not None and brick.id in bricks_pool:
        proper_pos = filled_bc.get_brick_position(brick)
        rot = brick.rotation
        if proper_pos[0] == placement_pos[0] - 1:
            rot = 180
        if proper_pos[2] == placement_pos[2] - 1:
            rot = 270
        return (brick.id, rot)
        #return labels[brick.id][rot]
    else:
        #return 0
        return ("None", 0)

def generate_samples_from_scene(bricks, filled_bc, gc, training_bc, config):
    samples = {subset: [] for subset in config['dataset']['subsets']}

    # look for the first position
    height_map = {subset: config['dataset']['subsets'][subset]['iteration']['height'] for subset in config['dataset']['subsets']}
    analyzed = {subset: np.zeros_like(training_bc.brick_grid) for subset in config['dataset']['subsets']}
    ntcs = find_next_pos_to_cover(gc, training_bc, analyzed, height_map)

    while any(x is not None for x in ntcs.values()):
        # choose subset used, positions and shape
        not_none_ntcs = dict((k, v) for k, v in ntcs.items() if v is not None)
        subset_used = list(not_none_ntcs.keys())[0]
        pos = not_none_ntcs[subset_used]
        for subs, indices in not_none_ntcs.items():
            if pos[1] < indices[1]:
                subset_used = subs
                pos = indices
        x, y, z = pos
        shape = np.array(config['dataset']['subsets'][subset_used]['iteration']['group_shape'])
        shape_top_ext = config['dataset']['subsets'][subset_used]['iteration']['shape_top_ext']
        placement_pos = np.array([x, y-1-shape_top_ext, z])
            
        looking_pos = np.array([x, y-1, z])
        shape_side_ext = int(np.round((shape[0]-1)/2))

        # get channels
        ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
        ext_vu_shape = ext_bu_to_vu(shape)
        channel1 = copy.deepcopy(gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                    ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                    ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]])

        channel2 = copy.deepcopy(training_bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                            ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                            ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]])

        # update training ext voxel grid and get label
        #label = None
        brick_id = "None"
        rotation = 0
        brick_variants = make_brick_variants(placement_pos, config['dataset']['subsets'][subset_used]['bricks'])
        if any(training_bc.is_placement_available(b) for b in brick_variants):
            brick_id, rotation = get_brick_id_rotation(filled_bc, looking_pos, placement_pos, config, subset_used, bricks)
            #label = get_label(filled_bc, looking_pos, placement_pos, config, subset_used, bricks)
            if brick_id != "None":
                place_brick(brick_id, rotation, placement_pos, training_bc)

        # append new sample
        samples[subset_used].append(sample_to_str(channel1, channel2, brick_id, rotation))

        # look for new position
        analyzed[subset_used][x, y, z] = True
        ntcs = find_next_pos_to_cover(gc, training_bc, analyzed, height_map)

    #print(f"samples generated:", len(samples))
    return samples

def make_samples(config):
    # initialize smaples dict
    all_samples = {subset: [] for subset in config['dataset']['subsets']}
    
    # load bricks
    filename = config['dataset']['raw_data_path']
    ldf = LDrawFile.load(filename)
    lbm = ldf.models[0]
    input_bricks = lbm.as_bricks()

    for k in range(4):
        bricks = utils_brick.rotate_bricks_y(input_bricks, k)

        # normalize bricks positions
        mins, _ = utils_brick.compute_bounds(bricks)
        for brick in bricks:
            brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([2, 4, 2])).astype(int))

        # prepare input data
        filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
        interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]
        gc = GeometryCoverage(interior_voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
        training_bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

        samples = generate_samples_from_scene(bricks, filled_bc, gc, training_bc, config)    
        for subset in all_samples:
            all_samples[subset].extend(samples[subset])
    return all_samples
