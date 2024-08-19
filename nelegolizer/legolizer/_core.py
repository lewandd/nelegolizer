import pyvista as pv
from nelegolizer import const
import numpy as np
from nelegolizer.data import LegoBrick
import nelegolizer.model.brick as brick
from torch import nn
from nelegolizer.utils import grid
import torch
from nelegolizer.model import brick_classification_models

fill_treshold = 0.1

def predictLegoBrick(*, voxel_grid: np.ndarray,
                        model: nn.Module, 
                        mesh_position: np.ndarray) -> LegoBrick:
  best_rotation = grid.find_best_rotation(voxel_grid)
  voxel_grid = grid.rotate(voxel_grid, best_rotation)

  fill_ratio = grid.get_fill_ratio(voxel_grid)
  if fill_ratio > fill_treshold:
    label = brick.test_predict(model, torch.tensor(voxel_grid).flatten())
    lego_brick = LegoBrick(label=label, mesh_position=mesh_position, rotation=best_rotation)
    return lego_brick
  else:
     return None
  
def check_subspace(*, voxel_grid: np.ndarray, 
                      position: tuple[int, int, int], 
                      shape: np.ndarray, 
                      LegoBrickList: list[LegoBrick]) -> None:
    voxel_subgrid = grid.get_subgrid(grid=voxel_grid, 
                                     position=position*const.BRICK_UNIT_RESOLUTION*shape, 
                                     shape=shape*const.BRICK_UNIT_RESOLUTION)
    mesh_position = np.array(position) * np.array(shape) * const.BRICK_UNIT_SHAPE

    if np.all(shape == (1, 1, 1)):
        lb = predictLegoBrick(voxel_grid=voxel_subgrid, 
                              model=brick_classification_models["model_n111"], 
                              mesh_position=mesh_position)
        if lb is not None:
            LegoBrickList.append(lb)

def legolize(path: str) -> list[LegoBrick]:    
    reader = pv.get_reader(path)
    mesh = reader.read()

    voxel_grid = grid.from_mesh(mesh, unit_shape=const.VOXEL_UNIT_SHAPE)
    voxel_grid = grid.extend(voxel_grid, required_dim_divisibility=const.BRICK_SHAPE_BOUND * const.BRICK_UNIT_RESOLUTION)

    LegoBrickList = []
    top_level_resolution = (voxel_grid.shape/(const.BRICK_SHAPE_BOUND * const.BRICK_UNIT_RESOLUTION)).astype(int)
    for position, _ in np.ndenumerate(np.zeros(top_level_resolution, dtype=LegoBrick)):
        check_subspace(voxel_grid=voxel_grid, position=position, shape=const.BRICK_SHAPE_BOUND, LegoBrickList=LegoBrickList)
    return LegoBrickList