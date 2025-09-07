import numpy as np
import pyvista as pv
import torch
from torch import nn
from typing import List, Tuple, Union

from nelegolizer import const
from nelegolizer.data import LegoBrick, ClassificationResult, ClassificationResult2
from nelegolizer.utils import grid
from nelegolizer.utils.conversion import *
from nelegolizer.utils import voxelization # noqa
#from nelegolizer.model import brick_classification_models
#from nelegolizer.data import part_by_size_label
import nelegolizer.model.brick as brick
from nelegolizer.model import shape_model_map, shape_model_map_cnn, rotate_division
from nelegolizer.model import initilize_models, initilize_models_cnn, initilize_models_csv
from nelegolizer.model.cnn import test_predict_cnn
from nelegolizer.data import initilize_parts

fill_treshold = 0.1

def voxelize(mesh: Union[str, pv.PolyData]):
    if isinstance(mesh, str):
        mesh_file_path = mesh
        reader = pv.get_reader(mesh_file_path)
        mesh = reader.read()
    elif not isinstance(mesh, pv.PolyData):
        raise ValueError("Legolize argument shold be either 3d object "
                         "path (str) or mesh (pyvista.PolyData)")
    mesh = mesh.flip_normal([0.0, 1.0, 0.0])

    voxel_grid = grid.from_mesh(mesh, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxel_grid = grid.provide_divisibility(
                                    voxel_grid,
                                    divider=bu_to_vu(const.LCH))
    return voxel_grid

def grid_regular_division(
                    voxel_grid: np.ndarray, 
                    vu_shape: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups_locations = []
    for position, _ in np.ndenumerate(voxel_grid[::vu_shape[0],
                                                 ::vu_shape[1],
                                                 ::vu_shape[2]]):
        shape = vu_to_bu(vu_shape)
        groups_locations.append((position*shape, shape))
    return groups_locations

def classify_group(position: np.ndarray,
                   shape: np.ndarray,
                   voxel_grid: np.ndarray) -> List[LegoBrick]:
    group = grid.get_subgrid(voxel_grid, bu_to_vu(position), bu_to_vu(shape) + 2*const.PADDING)
    if grid.is_empty(group):
        return []
    fill_ratio = grid.get_fill_ratio(group)
    if fill_ratio <= fill_treshold:
        return []
    
    rotation = grid.find_best_rotation(group)
    group = grid.rotate(group, rotation)

    model = shape_model_map[tuple(map(int, shape))]
    label = brick.test_predict(model, torch.tensor(group).flatten())[0]
    result = ClassificationResult(shape, label)
    if result.type == "brick":
        return [LegoBrick(id=str(result.brick_id), mesh_position=bu_to_mesh(position), rotation=rotation)]
    elif result.type == "division":
        #TODO test rotate_division function
        division = rotate_division(result.division, -rotation)
        _, offset1, shape1, offset2, shape2 = division
        lb_list1 = classify_group(position + offset1, shape1, voxel_grid)
        lb_list2 = classify_group(position + offset2, shape2, voxel_grid)
        return lb_list1 + lb_list2
    else: # result.type is None
        return []

def set_cover_grid(voxel_grid, bu):
  bu_shape = (np.array(voxel_grid.shape)/bu).astype(int)
  cover_grid = np.zeros(bu_shape, dtype=bool)
  for pos, _ in np.ndenumerate(cover_grid):
    x, y, z = pos
    cover_grid[x, y, z] = grid.get_fill(grid.get_subgrid(voxel_grid, bu_to_vu(pos), bu)) > 0
  return cover_grid

def find_next_to_cover(cover_grid, covered_grid):
  to_cover_mask = np.logical_and(cover_grid, np.logical_not(covered_grid))
  indices = np.argwhere(to_cover_mask)
  sorted_indices = indices[np.lexsort((indices[:,0], indices[:,2], indices[:,1]))]
  return sorted_indices[0] if len(sorted_indices) > 0 else None

def is_valid_placement(position, shape, covered_grid):
    return grid.is_empty(covered_grid[position[0]:position[0]+shape[0],
                                      position[1]:position[1]+shape[1],
                                      position[2]:position[2]+shape[2]])

def transpose(t):
    return (t[2],t[1],t[0])

def cover(position, shape, voxel_grid, cover_grid, covered_grid):
    # get group
    group = grid.get_subgrid(voxel_grid, bu_to_vu(position), bu_to_vu(shape) + 2*const.PADDING)

    # TODO: make group into two channels
    # - original as cover_group
    # - covered_grid as grid filled with bricks !!!

    # TODO: we need funcion to fill exact position with a lego brick voxel model
    # maybe just set_voxel_brick(position, brick_id, brick_rotation, grid) and
    # crate another bigger group with all neighbours bricks and then cut resize 
    # to smaller group ORRRRRR... another global 3d-array for covered in voxel
    # units and more details

    fill_ratio = grid.get_fill_ratio(group)
    if fill_ratio <= fill_treshold:
        covered_grid[position[0], position[1], position[2]] = True
        return []

    # classification
    group_rotation = 0
    shape_t = tuple(map(int, shape))
    if shape_t in shape_model_map:
        model = shape_model_map[shape_t]
        labels = brick.test_predict(model, torch.tensor(group).flatten())
    elif transpose(shape_t) in shape_model_map:
        group = grid.rotate(group, 90)
        group_rotation = 90
        model = shape_model_map[transpose(shape_t)]
        labels = brick.test_predict(model, torch.tensor(group).flatten())
    else:
        raise Exception(f"Got group shape {shape} with not network related to"
                        f"this group. Possible shapes: {shape_model_map.keys()}")
    
    results = [ClassificationResult2(shape, l) for l in labels]

    # iterate through results till brick or subspace found
    for result in results:
        # subspace was selected
        if result.type == "subshape":
            return cover(position, result.subshape, voxel_grid, cover_grid, covered_grid)
        # brick was selected
        elif result.type == "brick":
            global_rotation = result.rotation + group_rotation
            lb = LegoBrick(id=str(result.brick_id), 
                        mesh_position=bu_to_mesh(position), 
                        rotation=global_rotation)    
            
            global_shape = transpose(lb.part.size)

            # check if brick is valid
            if is_valid_placement(position, global_shape, covered_grid):
                covered_grid[position[0]:position[0]+global_shape.size[0], 
                             position[1]:position[1]+global_shape.size[1], 
                             position[2]:position[2]+global_shape.size[2]] = True
                return [lb]
        # nothing was selected for this location
        else:                 
            covered_grid[position[0], position[1], position[2]] = True
            return []

def cover2(pos, shape, oo, bo):
    shape_t = tuple(map(int, shape))
    input_shape = shape_t
    input_rotation = 0
    if shape_t not in shape_model_map_cnn:
        input_rotation = 90
        input_shape = transpose(shape_t)
    if input_shape not in shape_model_map_cnn:
        raise Exception(f"Got group shape {shape} with not network related to"
                        f"this group. Possible shapes: {shape_model_map_cnn.keys()}")

    vu_pos = bu_to_vu(pos)
    vu_shape = bu_to_vu(shape) + 2*const.PADDING
    
    channel1 = grid.get_subgrid(oo.voxel_grid, vu_pos, vu_shape)
    channel2 = grid.get_subgrid(bo.voxel_grid, vu_pos, vu_shape)

    if input_rotation == 90:
        channel1 = grid.rotate(channel1, 90)
        channel2 = grid.rotate(channel1, 90)

    # classification
    model = shape_model_map_cnn[input_shape]
    labels = test_predict_cnn(model, channel1, channel2)

    results = [ClassificationResult2(shape, l) for l in labels]

    # iterate through results till brick or subspace found
    for result in results:
        # subspace was selected
        if result.type == "subshape":
            return cover2(pos, result.subshape, oo, bo)
        # brick was selected
        elif result.type == "brick":
            #print(type(result.rotation))
            #print(type(input_rotation))
            brick_final_rotation = int(result.rotation) + input_rotation
            lb = LegoBrick(id=str(result.brick_id), 
                        mesh_position=bu_to_mesh(pos), 
                        rotation=brick_final_rotation)    
            
            #global_shape = transpose(lb.part.size)

            # check if brick is valid
            if is_valid_placement(pos, shape_t, bo.brick_grid):
                bo.place_brick(lb)
                return [lb]
        # nothing was selected for this location
        else:                 
            bo.brick_grid[pos[0], pos[1], pos[2]] = True
            return []

def legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
    initilize_parts()
    #initilize_models()
    initilize_models_csv()
    initilize_models_cnn()

    voxel_grid = voxelize(mesh)
    voxel_grid = grid.add_padding(voxel_grid, const.PADDING)

    oo = None#ObjectOccupancy(voxel_grid)
    bo = None#BrickOccupancy(oo.brick_grid.shape)

    #cover_grid = set_cover_grid(voxel_grid, const.BRICK_UNIT_RESOLUTION)
    #covered_grid = np.zeros_like(cover_grid)
    
    # LegoBrick output list
    bricks = []
    
    # fill all uncovered space
    next_to_cover = find_next_to_cover(oo.brick_grid, bo.brick_grid)
    #find_next_to_cover(cover_grid, covered_grid)

    # TODO należy przeglądać grupy o wszystkich rozmiarach
    # za każdym razem trzeba sprawdzać czy grupa się mieści w granicach
    # jeżeli się nie mieści, to należy może po prostu wtedy zastosować inną grupę
    # jeżeli to kraniec to nie ma wyboru jaką grupę wybrać, będzie tylko jedna możliwa
    # opcja albo rozmiar 2x1 dla jednej ściany albo 1x2 dla drugiej ściany albo 1x1 dla 3 ściany
    # może też być 1x1x1 dla sufitu, czyli po prostu jest if dla danego next_to_cover 
    # rozmiar poza granicą, to wtedy sprawdzamy kolejny rozmiar z listy
    # któryś będzie dobry

    while next_to_cover is not None:
        bricks += cover2(next_to_cover, const.LCH, oo, bo)
        #bricks += cover(next_to_cover, const.LCH, voxel_grid, cover_grid, covered_grid)
        next_to_cover = find_next_to_cover(oo.brick_grid, bo.brick_grid)
        #next_to_cover = find_next_to_cover(cover_grid, covered_grid)

    return bricks

def old_legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
    initilize_parts()
    initilize_models()

    voxel_grid = voxelize(mesh)
    cover_grid = set_cover_grid(voxel_grid, const.BRICK_UNIT_RESOLUTION)
    covered_grid = np.zeros_like(cover_grid)
    
    # add padding
    voxel_grid = grid.add_padding(voxel_grid, const.PADDING)
    
    # LegoBrick output list
    lb_list = []
    
    # fill all uncovered space
    next_to_cover = find_next_to_cover(cover_grid, covered_grid)
    while next_to_cover is not None:
        lb_list += cover(next_to_cover, const.LCH, voxel_grid, cover_grid, covered_grid)
        next_to_cover = find_next_to_cover(cover_grid, covered_grid)

    return lb_list