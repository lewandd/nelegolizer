from ..data import LDrawFile, LegoBrick, GeometryCoverage, LDrawFile, BrickCoverage, LegoBrick
from ..utils import brick as utils_brick
from ..legolizer.iterator import find_next_to_cover_net, place_brick
from ..utils.conversion import bu_to_mesh, ext_bu_to_vu
import numpy as np
import json

def sample_to_str(channel1: np.ndarray, channel2: np.ndarray, label: int) -> str:
    """
    Convert two 3D bool ndarrays and single int label into one-line str (JSON)
    """
    assert channel1.shape == channel2.shape, "Obie siatki muszą mieć ten sam wymiar"
    assert channel1.dtype == bool and channel2.dtype == bool, "Siatki powinny być typu bool"

    data_dict = {
        "channel1": channel1.astype(int).flatten().tolist(),
        "channel2": channel2.astype(int).flatten().tolist(),
        "shape": channel1.shape,
        "label": int(label)
    }
    return json.dumps(data_dict)

def save_dataset(samples: list, filename: str):
    """
    Save samples to txt file.
    """
    with open(filename, "w") as f:
        for s in samples:
            f.write(s + "\n")

shape_id_rot_label_map = {(2, 3, 2): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:1, 90:2, 180:1, 270:2}},
                          (2, 3, 1): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:2, 90:1, 180:2, 270:1}},
                          (1, 3, 1): {"3005": {0:2, 90:2, 180:2, 270:2},
                                      "54200": {0:3, 90:4, 180:5, 270:6},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:2, 90:2, 180:2, 270:2}},
                          (1, 1, 1): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:1, 90:1, 180:1, 270:1}}}

def make_brick_variants(placement_pos, network_type):
    if network_type == 3:
        return {"3004_0": LegoBrick(id="3004", mesh_position=bu_to_mesh(placement_pos), rotation=0),
                "3004_90": LegoBrick(id="3004", mesh_position=bu_to_mesh(placement_pos), rotation=90),
                "3004_180": LegoBrick(id="3004", mesh_position=bu_to_mesh(placement_pos - np.array([1, 0, 0])), rotation=180),
                "3004_270": LegoBrick(id="3004", mesh_position=bu_to_mesh(placement_pos - np.array([0, 0, 1])), rotation=270),
                "3005": LegoBrick(id="3005", mesh_position=bu_to_mesh(placement_pos), rotation=0)}
    elif network_type == 1: 
        return {"3024_0": LegoBrick(id="3024", mesh_position=bu_to_mesh(placement_pos), rotation=0)}

labels = {3: {"3004": {0: 1, 90: 2, 180: 3, 270: 4},
              "3005": {0: 5, 90: 5, 180: 5, 270: 5},
              "54200":{0: 6, 90: 7, 180: 8, 270: 9}},
          1: {"3024": {0: 1, 90: 1, 180: 1, 270: 1},}}

def get_label(filled_bc, looking_pos, placement_pos, network_type, bricks):
    brick = filled_bc.get_brick_at(bricks, looking_pos)
    if brick is not None and brick.id in labels[network_type].keys():
        proper_pos = filled_bc.get_brick_position(brick)
        rot = brick.rotation
        if proper_pos[0] == placement_pos[0] - 1:
            rot = 180
        if proper_pos[2] == placement_pos[2] - 1:
            rot = 270
        return labels[network_type][brick.id][rot]
    else:
        return 0

def make_samples(filename, samples_network_type):
    ldf = LDrawFile.load(filename)
    lbm = ldf.models[0]
    bricks = lbm.as_bricks()

    mins, _ = utils_brick.compute_bounds(bricks)

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

    samples = []

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
        label = None
        placed = False
        if any(training_bc.is_placement_available(b) 
            for b in make_brick_variants(placement_pos, network_type).values()):
            label = get_label(filled_bc, looking_pos, placement_pos, network_type, bricks)
            
            if label != 0:
                placed = place_brick(label, placement_pos, network_type, training_bc)

        
        analyzed[network_type][x, y, z] = True
        ntc1, ntc3 = find_next_to_cover_net(gc, training_bc, analyzed)

        ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
        ext_vu_shape = ext_bu_to_vu(shape)

        channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                    ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                    ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

        channel2 = training_bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                            ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                            ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]
        if label is not None and samples_network_type == network_type:
            samples.append(sample_to_str(channel1, channel2, label))

    print(f"samples generated:", len(samples))
    return samples

