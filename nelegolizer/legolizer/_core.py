import numpy as np
import pyvista as pv
from typing import List, Union

from ..constants import BU_RES, VU
from ..data import LegoBrick, initilize_parts, GeometryCoverage, BrickCoverage, LDrawModel
from ..utils import voxelization as utils_voxelization
from ..utils import grid as utils_grid
from .iterator import find_next_pos_to_cover, make_brick_variants, place_brick
import yaml
from ..paths import DEFAULT_CONFIG
from ..model.inference import predict
from ..model.registry import get_model
from ..model.cnn import *
from ..utils.conversion import ext_bu_to_vu, bu_to_mesh
from ..model.label_encoder import build_label_encoder
from ..data import LDrawFile
from ..utils import brick as utils_brick
from ..model.evaluation import compute_iou, compute_stability_cost

#fill_treshold = 0.1

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def legolize_from_mpd(filepath: str, legolize_config_path: str = DEFAULT_CONFIG, visualize: bool = True) -> List[LegoBrick]:
    # load mpd file
    ldf = LDrawFile.load(filepath)
    lbm = LDrawModel.merge_multiple_models(ldf.models)
    bricks = lbm.as_bricks()

    # normalize bricks positions
    mins, _ = utils_brick.compute_bounds(bricks)
    for brick in bricks:
        brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([2, 4, 2])).astype(int))

    # prepare input data
    filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
    interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]
    voxel_grid = interior_voxel_grid
    gc = GeometryCoverage(voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
    bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

    # load configs
    legolize_config = load_config(legolize_config_path)
    model_classes = {name: load_config(m['config_path']) for name, m in legolize_config['legolization']['models'].items()}
    model_instances = {name: get_model(conf) for name, conf in model_classes.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, m in model_instances.items():
        m.load_state_dict(torch.load(legolize_config['legolization']['models'][name]['model_path'], map_location=device))    
    
    # prepare for classification
    label_encoders = {net: build_label_encoder(conf) for net, conf in model_classes.items()}
    height_map = {net: conf['iteration']['height'] for net, conf in model_classes.items()}
    analyzed = {net: np.zeros_like(bc.brick_grid) for net in model_classes}
    ntcs = find_next_pos_to_cover(gc, bc, analyzed, height_map)
    bricks = []

    while any(x is not None for x in ntcs.values()):
        # choose subset used, positions and shape
        not_none_ntcs = dict((k, v) for k, v in ntcs.items() if v is not None)
        net_used = list(not_none_ntcs.keys())[0]
        pos = not_none_ntcs[net_used]
        for subs, indices in not_none_ntcs.items():
            if pos[1] < indices[1]:
                net_used = subs
                pos = indices
        model_config = model_classes[net_used]['model']
        iter_config = model_classes[net_used]['iteration']
        
        x, y, z = pos
        shape = np.array(iter_config['group_shape'])
        shape_top_ext = iter_config['shape_top_ext']
        placement_pos = np.array([x, y-1-shape_top_ext, z])
            
        shape_side_ext = int(np.round((shape[0]-1)/2))

        brick_variants = make_brick_variants(placement_pos, model_config['bricks'])
        if any(bc.is_placement_available(b) for b in brick_variants):
            ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
            ext_vu_shape = ext_bu_to_vu(shape)

            # get channels
            channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                         ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                         ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

            channel2 = bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                         ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                         ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

            model = model_instances[net_used]
            predict_list = predict(model, channel1, channel2)
            placed = False
            label = 100000
            it = 0
            while label != 0 and not placed:
                label = predict_list[it]
                brick_id, rotation = label_encoders[net_used].decode(label)
                if brick_id != "None":
                    new_brick = place_brick(brick_id, rotation, placement_pos, bc)
                    if new_brick is not None:
                        bricks.append(new_brick)
                        placed = True
                it += 1

        # look for new position
        analyzed[net_used][x, y, z] = True
        ntcs = find_next_pos_to_cover(gc, bc, analyzed, height_map)

    if visualize:
        voxels1 = utils_voxelization.from_grid(filled_bc.ext_voxel_grid, voxel_mesh_shape=VU)
        voxels2 = utils_voxelization.from_grid(bc.ext_voxel_grid, voxel_mesh_shape=VU)
        plotter = pv.Plotter(shape=(1, 2))
        plotter.subplot(0, 0)
        plotter.add_title("filled", 8)
        plotter.add_mesh(voxels1)

        plotter.subplot(0, 1)
        plotter.add_title("classified", 8)
        plotter.add_mesh(voxels2)
        plotter.show()

    print(f"IoU: {compute_iou(filled_bc.ext_voxel_grid, bc.ext_voxel_grid)}")
    print(f"SC: {compute_stability_cost(bricks)}")

    return bricks

def legolize(mesh: Union[str, pv.PolyData], legolize_config_path: str = DEFAULT_CONFIG, visualize: bool = True) -> List[LegoBrick]:
    initilize_parts()

    # prepare input data
    voxel_grid = utils_voxelization.voxelize(mesh)
    voxel_grid = utils_grid.provide_divisibility(voxel_grid, divider=BU_RES)
    gc = GeometryCoverage(voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
    bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

    # load configs
    legolize_config = load_config(legolize_config_path)
    model_classes = {name: load_config(m['config_path']) for name, m in legolize_config['legolization']['models'].items()}
    model_instances = {name: get_model(conf) for name, conf in model_classes.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, m in model_instances.items():
        m.load_state_dict(torch.load(legolize_config['legolization']['models'][name]['model_path'], map_location=device))  

    # prepare for classification
    label_encoders = {net: build_label_encoder(conf) for net, conf in model_classes.items()}
    height_map = {net: conf['iteration']['height'] for net, conf in model_classes.items()}
    analyzed = {net: np.zeros_like(bc.brick_grid) for net in model_classes}
    ntcs = find_next_pos_to_cover(gc, bc, analyzed, height_map)
    bricks = []

    while any(x is not None for x in ntcs.values()):
        # choose subset used, positions and shape
        not_none_ntcs = dict((k, v) for k, v in ntcs.items() if v is not None)
        net_used = list(not_none_ntcs.keys())[0]
        pos = not_none_ntcs[net_used]
        for subs, indices in not_none_ntcs.items():
            if pos[1] < indices[1]:
                net_used = subs
                pos = indices
        model_config = model_classes[net_used]['model']
        iter_config = model_classes[net_used]['iteration']
        
        x, y, z = pos
        shape = np.array(iter_config['group_shape'])
        shape_top_ext = iter_config['shape_top_ext']
        placement_pos = np.array([x, y-1-shape_top_ext, z])
            
        shape_side_ext = int(np.round((shape[0]-1)/2))

        brick_variants = make_brick_variants(placement_pos, model_config['bricks'])
        if any(bc.is_placement_available(b) for b in brick_variants):
            ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
            ext_vu_shape = ext_bu_to_vu(shape)

            # get channels
            channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                         ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                         ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

            channel2 = bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                         ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                         ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

            model = model_instances[net_used]
            predict_list = predict(model, channel1, channel2)
            placed = False
            label = 100000
            it = 0
            while label != 0 and not placed:
                label = predict_list[it]
                brick_id, rotation = label_encoders[net_used].decode(label)
                if brick_id != "None":
                    new_brick = place_brick(brick_id, rotation, placement_pos, bc)
                    if new_brick is not None:
                        bricks.append(new_brick)
                        placed = True
                it += 1

        # look for new position
        analyzed[net_used][x, y, z] = True
        ntcs = find_next_pos_to_cover(gc, bc, analyzed, height_map)

    if visualize:
        voxels1 = utils_voxelization.from_grid(gc.ext_voxel_grid, voxel_mesh_shape=VU)
        voxels2 = utils_voxelization.from_grid(bc.ext_voxel_grid, voxel_mesh_shape=VU)
        plotter = pv.Plotter(shape=(1, 2))
        plotter.subplot(0, 0)
        plotter.add_title("original 3d obj", 8)
        plotter.add_mesh(voxels1)

        plotter.subplot(0, 1)
        plotter.add_title("classified", 8)
        plotter.add_mesh(voxels2)
        plotter.show()

    return bricks
