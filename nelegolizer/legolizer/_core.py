import numpy as np
import pyvista as pv
from typing import List, Union

from ..constants import BU_RES, VU
from ..data import LegoBrick, initilize_parts, GeometryCoverage, BrickCoverage
from ..utils import voxelization as utils_voxelization
from ..utils import grid as utils_grid
from .iterator import find_next_pos_to_cover, make_brick_variants, place_brick
import yaml
from ..paths import MODEL555CONFIG, MODEL555
from ..model.inference import predict
from ..model.registry import get_model
from ..model.cnn import *
from ..utils.conversion import ext_bu_to_vu, bu_to_mesh
from ..model.label_encoder import build_label_encoder
from ..data import LDrawFile
from ..utils import brick as utils_brick

#fill_treshold = 0.1

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
    initilize_parts()

    filename = "../../data/raw/ldraw_models/simple_mountain.mpd"
    ldf = LDrawFile.load(filename)
    lbm = ldf.models[0]
    bricks = lbm.as_bricks()

    # normalize bricks positions
    mins, _ = utils_brick.compute_bounds(bricks)
    for brick in bricks:
        brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+np.array([2, 4, 2])).astype(int))

    # prepare input data
    filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
    interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]

    voxel_grid = utils_voxelization.voxelize(mesh)
    voxel_grid = utils_grid.provide_divisibility(voxel_grid, divider=BU_RES)
    #voxel_grid = interior_voxel_grid

    gc = GeometryCoverage(voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
    bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

    config = {'model555': load_config(MODEL555CONFIG)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model555 = get_model(load_config(MODEL555CONFIG))
    model555.load_state_dict(torch.load(MODEL555, map_location=device))
    
    models = {'model555': model555}
    label_encoders = {net: build_label_encoder(conf) for net, conf in config.items()}
    height_map = {net: conf['iteration']['height'] for net, conf in config.items()}
    analyzed = {net: np.zeros_like(bc.brick_grid) for net in config}
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
        model_config = config[net_used]['model']
        iter_config = config[net_used]['iteration']
        
        x, y, z = pos
        shape = np.array(iter_config['group_shape'])
        shape_top_ext = iter_config['shape_top_ext']
        placement_pos = np.array([x, y-1-shape_top_ext, z])
            
        looking_pos = np.array([x, y-1, z])
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

            placed = False
            label = 100000
            it = 0
            while label != 0 and not placed:
                # model
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #model = get_model(nets[net_used])
                #model.load_state_dict(torch.load(models[net_used], map_location=device))
                model = models[net_used]
                label = predict(model, channel1, channel2)[it]
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

    return bricks

#    voxels = utils_voxelization.from_grid(bc.ext_voxel_grid, voxel_mesh_shape=VU)
#    plotter = pv.Plotter()
#    plotter.add_mesh(voxels)
#    plotter.show()

    #voxel_grid = grid.add_padding(voxel_grid, 0)

    #oo = None#ObjectOccupancy(voxel_grid)
    #bo = None#BrickOccupancy(oo.brick_grid.shape)

    #cover_grid = set_cover_grid(voxel_grid, BU_RES)
    #covered_grid = np.zeros_like(cover_grid)
    
    # LegoBrick output list
    #bricks = []
    
    # fill all uncovered space
    #next_to_cover = find_next_to_cover(oo.brick_grid, bo.brick_grid)
    #find_next_to_cover(cover_grid, covered_grid)

    # TODO należy przeglądać grupy o wszystkich rozmiarach
    # za każdym razem trzeba sprawdzać czy grupa się mieści w granicach
    # jeżeli się nie mieści, to należy może po prostu wtedy zastosować inną grupę
    # jeżeli to kraniec to nie ma wyboru jaką grupę wybrać, będzie tylko jedna możliwa
    # opcja albo rozmiar 2x1 dla jednej ściany albo 1x2 dla drugiej ściany albo 1x1 dla 3 ściany
    # może też być 1x1x1 dla sufitu, czyli po prostu jest if dla danego next_to_cover 
    # rozmiar poza granicą, to wtedy sprawdzamy kolejny rozmiar z listy
    # któryś będzie dobry

    #while next_to_cover is not None:
    #    bricks += cover2(next_to_cover, LCH, oo, bo)
    #    #bricks += cover(next_to_cover, LCH, voxel_grid, cover_grid, covered_grid)
    #    next_to_cover = find_next_to_cover(oo.brick_grid, bo.brick_grid)
        #next_to_cover = find_next_to_cover(cover_grid, covered_grid)

    #return bricks

#def grid_regular_division(
#                    voxel_grid: np.ndarray, 
#                    vu_shape: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
#    groups_locations = []
#    for position, _ in np.ndenumerate(voxel_grid[::vu_shape[0],
#                                                 ::vu_shape[1],
#                                                 ::vu_shape[2]]):
#        shape = vu_to_bu(vu_shape)
#        groups_locations.append((position*shape, shape))
#    return groups_locations

#def classify_group(position: np.ndarray,
#                   shape: np.ndarray,
#                   voxel_grid: np.ndarray) -> List[LegoBrick]:
#    group = grid.get_subgrid(voxel_grid, bu_to_vu(position), bu_to_vu(shape))
#    if grid.is_empty(group):
#        return []
#    fill_ratio = grid.get_fill_ratio(group)
#    if fill_ratio <= fill_treshold:
#        return []
    
#    rotation = grid.find_best_rotation(group)
#    group = grid.rotate(group, rotation)

#    model = shape_model_map[tuple(map(int, shape))]
#    label = brick.test_predict(model, torch.tensor(group).flatten())[0]
#    result = ClassificationResult(shape, label)
#    if result.type == "brick":
#        return [LegoBrick(id=str(result.brick_id), mesh_position=bu_to_mesh(position), rotation=rotation)]
#    elif result.type == "division":
        #TODO test rotate_division function
#        division = rotate_division(result.division, -rotation)
#        _, offset1, shape1, offset2, shape2 = division
#        lb_list1 = classify_group(position + offset1, shape1, voxel_grid)
#        lb_list2 = classify_group(position + offset2, shape2, voxel_grid)
#        return lb_list1 + lb_list2
#    else: # result.type is None
#        return []

#def set_cover_grid(voxel_grid, bu):
#  bu_shape = (np.array(voxel_grid.shape)/bu).astype(int)
#  cover_grid = np.zeros(bu_shape, dtype=bool)
#  for pos, _ in np.ndenumerate(cover_grid):
#    x, y, z = pos
#    cover_grid[x, y, z] = grid.get_fill(grid.get_subgrid(voxel_grid, bu_to_vu(pos), bu)) > 0
#  return cover_grid

#def find_next_to_cover(cover_grid, covered_grid):
#  to_cover_mask = np.logical_and(cover_grid, np.logical_not(covered_grid))
#  indices = np.argwhere(to_cover_mask)
#  sorted_indices = indices[np.lexsort((indices[:,0], indices[:,2], indices[:,1]))]
#  return sorted_indices[0] if len(sorted_indices) > 0 else None

#def is_valid_placement(position, shape, covered_grid):
#    return grid.is_empty(covered_grid[position[0]:position[0]+shape[0],
#                                      position[1]:position[1]+shape[1],
#                                      position[2]:position[2]+shape[2]])

#def transpose(t):
#    return (t[2],t[1],t[0])

#def cover(position, shape, voxel_grid, cover_grid, covered_grid):
    # get group
#    group = grid.get_subgrid(voxel_grid, bu_to_vu(position), bu_to_vu(shape))

    # TODO: make group into two channels
    # - original as cover_group
    # - covered_grid as grid filled with bricks !!!

    # TODO: we need funcion to fill exact position with a lego brick voxel model
    # maybe just set_voxel_brick(position, brick_id, brick_rotation, grid) and
    # crate another bigger group with all neighbours bricks and then cut resize 
    # to smaller group ORRRRRR... another global 3d-array for covered in voxel
    # units and more details

#    fill_ratio = grid.get_fill_ratio(group)
#    if fill_ratio <= fill_treshold:
#        covered_grid[position[0], position[1], position[2]] = True
#        return []

    # classification
#    group_rotation = 0
#    shape_t = tuple(map(int, shape))
#    if shape_t in shape_model_map:
#        model = shape_model_map[shape_t]
#        labels = brick.test_predict(model, torch.tensor(group).flatten())
#    elif transpose(shape_t) in shape_model_map:
#        group = grid.rotate(group, 90)
#        group_rotation = 90
#        model = shape_model_map[transpose(shape_t)]
#        labels = brick.test_predict(model, torch.tensor(group).flatten())
#    else:
#        raise Exception(f"Got group shape {shape} with not network related to"
#                        f"this group. Possible shapes: {shape_model_map.keys()}")
    
#    results = [ClassificationResult2(shape, l) for l in labels]

    # iterate through results till brick or subspace found
#    for result in results:
        # subspace was selected
#        if result.type == "subshape":
#            return cover(position, result.subshape, voxel_grid, cover_grid, covered_grid)
        # brick was selected
#        elif result.type == "brick":
#            global_rotation = result.rotation + group_rotation
#            lb = LegoBrick(id=str(result.brick_id), 
#                        mesh_position=bu_to_mesh(position), 
#                        rotation=global_rotation)    
            
#            global_shape = transpose(lb.part.size)

            # check if brick is valid
#            if is_valid_placement(position, global_shape, covered_grid):
#                covered_grid[position[0]:position[0]+global_shape.size[0], 
#                             position[1]:position[1]+global_shape.size[1], 
#                             position[2]:position[2]+global_shape.size[2]] = True
#                return [lb]
        # nothing was selected for this location
#        else:                 
#            covered_grid[position[0], position[1], position[2]] = True
#            return []

#def cover2(pos, shape, oo, bo):
#    shape_t = tuple(map(int, shape))
#    input_shape = shape_t
#    input_rotation = 0
#    if shape_t not in shape_model_map_cnn:
#        input_rotation = 90
#        input_shape = transpose(shape_t)
#    if input_shape not in shape_model_map_cnn:
#        raise Exception(f"Got group shape {shape} with not network related to"
#                        f"this group. Possible shapes: {shape_model_map_cnn.keys()}")

#    vu_pos = bu_to_vu(pos)
#    vu_shape = bu_to_vu(shape)
    
#    channel1 = grid.get_subgrid(oo.voxel_grid, vu_pos, vu_shape)
#    channel2 = grid.get_subgrid(bo.voxel_grid, vu_pos, vu_shape)

#    if input_rotation == 90:
#        channel1 = grid.rotate(channel1, 90)
#        channel2 = grid.rotate(channel1, 90)

    # classification
#    model = shape_model_map_cnn[input_shape]
#    labels = test_predict_cnn(model, channel1, channel2)

#    results = [ClassificationResult2(shape, l) for l in labels]

    # iterate through results till brick or subspace found
#    for result in results:
        # subspace was selected
#        if result.type == "subshape":
#            return cover2(pos, result.subshape, oo, bo)
        # brick was selected
#        elif result.type == "brick":
            #print(type(result.rotation))
            #print(type(input_rotation))
#            brick_final_rotation = int(result.rotation) + input_rotation
#            lb = LegoBrick(id=str(result.brick_id), 
#                        mesh_position=bu_to_mesh(pos), 
#                        rotation=brick_final_rotation)    
            
            #global_shape = transpose(lb.part.size)

            # check if brick is valid
#            if is_valid_placement(pos, shape_t, bo.brick_grid):
#                bo.place_brick(lb)
#                return [lb]
        # nothing was selected for this location
#        else:                 
#            bo.brick_grid[pos[0], pos[1], pos[2]] = True
#            return []

#def old_legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
#    initilize_parts()
#    initilize_models()

#    voxel_grid = voxelize(mesh)
#    cover_grid = set_cover_grid(voxel_grid, BU_RES)
#    covered_grid = np.zeros_like(cover_grid)
    
    # add padding
#    voxel_grid = grid.add_padding(voxel_grid, 0)
    
    # LegoBrick output list
#    lb_list = []
    
    # fill all uncovered space
#    next_to_cover = find_next_to_cover(cover_grid, covered_grid)
#    while next_to_cover is not None:
#        lb_list += cover(next_to_cover, LCH, voxel_grid, cover_grid, covered_grid)
#        next_to_cover = find_next_to_cover(cover_grid, covered_grid)

#    return lb_list