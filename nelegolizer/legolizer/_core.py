import numpy as np
import pyvista as pv
import torch
from torch import nn
from typing import List, Tuple, Union

from nelegolizer import const
from nelegolizer.data import LegoBrick, ClassificationResult
from nelegolizer.utils import grid
from nelegolizer.utils.grid import vu_to_bu, bu_to_vu, bu_to_mesh
from nelegolizer.utils import voxelization # noqa
#from nelegolizer.model import brick_classification_models
#from nelegolizer.data import part_by_size_label
import nelegolizer.model.brick as brick
from nelegolizer.model import shape_model_map, rotate_division

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
    label = brick.test_predict(model, torch.tensor(group).flatten())
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

def legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
    voxel_grid = voxelize(mesh)
    groups_locations = grid_regular_division(voxel_grid, 
                                             vu_shape=bu_to_vu(const.LCH))
    voxel_grid = grid.add_padding(voxel_grid, const.PADDING)

    lb_list = []
    for gl in groups_locations:
        position, shape = gl
        lb_list += classify_group(
                       position=position,
                       shape=shape,
                       voxel_grid=voxel_grid)
    return lb_list