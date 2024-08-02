import pyvista as pv
from nelegolizer.constants import BRICK_UNIT_SHAPE, VOXEL_UNIT_SHAPE, BRICK_SHAPE_BOUND, BRICK_UNIT_RESOLUTION, BRICK_SHAPES
import numpy as np
from nelegolizer.data import LegoBrick
import nelegolizer.model.object as obj
from torch import nn
import nelegolizer
from nelegolizer.utils import grid
import torch

fill_treshold = 0.1

def predictLegoBrick(*, voxel_grid: np.ndarray,
                        model: nn.Module, 
                        mesh_position: np.ndarray) -> LegoBrick:
  best_rotation = grid.find_best_rotation(voxel_grid)
  voxel_grid = grid.rotate(voxel_grid, best_rotation)

  fill_ratio = grid.get_fill_ratio(voxel_grid)
  if fill_ratio > fill_treshold:
    label = obj.test_predict(model, torch.tensor(voxel_grid).flatten())
    lego_brick = LegoBrick(label=label, mesh_position=mesh_position, rotation=best_rotation)
    return lego_brick
  else:
     return None
  
def check_subspace(*, voxel_grid: np.ndarray, 
                      position: tuple[int, int, int], 
                      shape: np.ndarray, 
                      LegoBrickList: list[LegoBrick]) -> None:
    voxel_subgrid = grid.get_subgrid(grid=voxel_grid, 
                                     position=position*BRICK_UNIT_RESOLUTION*shape, 
                                     shape=shape*BRICK_UNIT_RESOLUTION)
    mesh_position = np.array(position) * np.array(shape) * BRICK_UNIT_SHAPE

    if np.all(shape == (1, 1, 1)):
        LegoBrickList.append(predictLegoBrick(voxel_grid=voxel_subgrid, 
                                              model=nelegolizer.model.models["model_n111"], 
                                              mesh_position=mesh_position))

def legolize(path: str) -> list[LegoBrick]:    
    reader = pv.get_reader(path)
    mesh = reader.read()

    voxel_grid = grid.from_mesh(mesh, 
                                unit_shape=VOXEL_UNIT_SHAPE, 
                                required_dim_divisibility=BRICK_SHAPE_BOUND * BRICK_UNIT_RESOLUTION)

    LegoBrickList = []
    upper_level_resolution = (voxel_grid.shape/(BRICK_SHAPE_BOUND * BRICK_UNIT_RESOLUTION)).astype(int)
    for position, _ in np.ndenumerate(np.zeros(upper_level_resolution, dtype=LegoBrick)):
        check_subspace(voxel_grid=voxel_grid, position=position, shape=BRICK_SHAPE_BOUND, LegoBrickList=LegoBrickList)
    return LegoBrickList