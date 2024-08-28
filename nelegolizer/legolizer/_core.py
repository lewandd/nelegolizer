import numpy as np
import pyvista as pv
import torch
from torch import nn
from typing import List, Tuple

from nelegolizer import const
from nelegolizer.data import LegoBrick
from nelegolizer.utils import grid
from nelegolizer.model import brick_classification_models
import nelegolizer.model.brick as brick

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
                              mesh_position=mesh_position)
        if lb is not None:
            LegoBrickList.append(lb)

def legolize(mesh: str | pv.PolyData) -> List[LegoBrick]:    
    if isinstance(mesh, str):
        mesh_file_path = mesh
        reader = pv.get_reader(mesh_file_path)
        mesh = reader.read()
    elif not isinstance(mesh, pv.PolyData):
        raise ValueError("Legolize argument shold be either 3d object path (str) or mesh (pyvista.PolyData)")
    mesh = mesh.flip_normal([0.0, 1.0, 0.0])

    voxel_grid = grid.from_mesh(mesh, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxel_grid = grid.provide_divisibility(voxel_grid, divider=const.TOP_LEVEL_BRICK_RESOLUTION)

    LegoBrickList = []
    top_level_grid_resolution = (voxel_grid.shape/(const.TOP_LEVEL_BRICK_RESOLUTION)).astype(int)
    
    voxel_grid = grid.add_padding(voxel_grid, const.PADDING)
    for position, _ in np.ndenumerate(np.zeros(top_level_grid_resolution, dtype=LegoBrick)):
        check_subspace(voxel_grid=voxel_grid, position=position, shape=const.TOP_LEVEL_BRICK_SHAPE, LegoBrickList=LegoBrickList)
    return LegoBrickList