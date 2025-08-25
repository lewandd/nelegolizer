import numpy as np
import pyvista as pv
import torch
from torch import nn
from typing import List, Tuple, Union

from nelegolizer import const
from nelegolizer.data import LegoBrick
from nelegolizer.utils import grid
from nelegolizer.utils.grid import vu_to_bu, bu_to_vu
from nelegolizer.utils import voxelization # noqa
from nelegolizer.model import brick_classification_models
from nelegolizer.data import part_by_size_label
import nelegolizer.model.brick as brick

fill_treshold = 0.1


def predictLegoBrick(*, voxel_grid: np.ndarray,
                     model: nn.Module,
                     shape: np.ndarray,
                     mesh_position: np.ndarray) -> LegoBrick:
    best_rotation = grid.find_best_rotation(voxel_grid)
    voxel_grid = grid.rotate(voxel_grid, best_rotation)

    fill_ratio = grid.get_fill_ratio(voxel_grid)
    if fill_ratio > fill_treshold:
        # mesh = voxelization.from_grid(voxel_grid,
        #                              voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
        # p = pv.Plotter()
        # p.add_mesh(mesh)
        # p.show_bounds(bounds=[0.0, 0.9, 0.0, 1.2, 0.0, 0.9])
        # p.show(cpos="xy")
        label = brick.test_predict(model, torch.tensor(voxel_grid).flatten())
        shape_tuple = tuple(map(int, shape))
        id = part_by_size_label[str(shape_tuple)][label].brick_id
        lego_brick = LegoBrick(id=id,
                               mesh_position=mesh_position,
                               rotation=best_rotation)
        return lego_brick
    else:
        return None


def check_subspace(*, voxel_grid: np.ndarray,
                   position: Tuple[int, int, int],
                   shape: np.ndarray,
                   LegoBrickList: List[LegoBrick]) -> None:
    position = np.array(position)
    voxel_shape = shape * const.BRICK_UNIT_RESOLUTION
    voxel_subgrid = grid.get_subgrid(grid=voxel_grid,
                                     position=position*voxel_shape,
                                     shape=voxel_shape + 2*const.PADDING)
    mesh_shape = shape * const.BRICK_UNIT_MESH_SHAPE
    mesh_position = position * mesh_shape

    if np.all(shape == (1, 1, 1)):
        lb = predictLegoBrick(voxel_grid=voxel_subgrid,
                              model=brick_classification_models["model_n111"],
                              shape=shape,
                              mesh_position=mesh_position)
        if lb is not None:
            LegoBrickList.append(lb)


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
        groups_locations.append((position , shape))
    return groups_locations


def legolize(mesh: Union[str, pv.PolyData]) -> List[LegoBrick]:
    voxel_grid = voxelize(mesh)
    groups_locations = grid_regular_division(voxel_grid, 
                                             vu_shape=bu_to_vu(const.LCH))
    LegoBrickList = []
    voxel_grid = grid.add_padding(voxel_grid, const.PADDING)

    for gl in groups_locations:
        position, shape = gl
        check_subspace(voxel_grid=voxel_grid,
                       position=position,
                       shape=shape,
                       LegoBrickList=LegoBrickList)
    return LegoBrickList
