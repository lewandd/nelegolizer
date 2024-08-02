import pyvista as pv
from nelegolizer.constants import BRICK_UNIT_SHAPE, VOXEL_UNIT_SHAPE, BRICK_SHAPE_BOUND, BRICK_UNIT_RESOLUTION, BRICK_SHAPES
import numpy as np
from nelegolizer.data import LegoBrick
import nelegolizer.model.object as obj
from torch import nn
import nelegolizer
from nelegolizer.utils import grid
from nelegolizer.utils import voxelization
import torch

fill_treshold = 0.1

def predictLegoBrick(*, 
                  voxel_grid: np.ndarray,
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
  
def check_subspace(*,
                   voxel_subgrid: np.ndarray, 
                   position: tuple[int, int, int], 
                   shape: np.ndarray, 
                   LegoBrickGrid: list[list[list[list[LegoBrick]]]]):
    x, y, z = position
    mesh_position = np.array(position) * np.array(shape) * BRICK_UNIT_SHAPE

    if np.all(shape == (1, 1, 1)):
        LegoBrickGrid[str(shape)][x][y][z] = predictLegoBrick(voxel_grid=voxel_subgrid, 
                                                  model=nelegolizer.model.models["model_n111"], 
                                                  mesh_position=mesh_position)

def legolize(path):    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    pv_voxels = voxelization.from_mesh(mesh, unit_shape=VOXEL_UNIT_SHAPE)
    voxel_grid = grid.from_pv_voxels(pv_voxels,
                                     unit_shape=VOXEL_UNIT_SHAPE, 
                                     required_dim_divisibility=BRICK_SHAPE_BOUND * BRICK_UNIT_RESOLUTION)

    # create LegoBrickGrid dictionary of LegoBrick
    LegoBrickGrid = {}
    for shape in BRICK_SHAPES:
        LegoBrickGrid[str(shape)] = np.zeros((voxel_grid.shape/(BRICK_UNIT_RESOLUTION * shape)).astype(int), dtype=LegoBrick)   

    for shape in BRICK_SHAPES:
        for (i, j, k), _ in np.ndenumerate(LegoBrickGrid[str(shape)]):
            start = BRICK_UNIT_RESOLUTION * shape * np.array([i, j, k])
            end =   BRICK_UNIT_RESOLUTION * shape * np.array([i+1, j+1, k+1])
            voxel_subgrid = voxel_grid[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            check_subspace(voxel_subgrid=voxel_subgrid, position=(i, j, k), shape=shape, LegoBrickGrid=LegoBrickGrid)
    return LegoBrickGrid