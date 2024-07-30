import pyvista as pv
import nelegolizer.constants as CONST
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
                   shape: tuple[int, int, int], 
                   LegoBrickGrid: list[list[list[list[LegoBrick]]]]):
    x, y, z = position
    mesh_position = np.array(position) * np.array(shape) * np.array([0.8, 1.12, 0.8])

    if shape == (1, 1, 1):
        LegoBrickGrid[0][x][y][z] = predictLegoBrick(voxel_grid=voxel_subgrid, 
                                                  model=nelegolizer.model.models["model_n111"], 
                                                  mesh_position=mesh_position)

def legolize(path, target_res):
    RESOLUTION = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    pv_voxels = voxelization.from_mesh(mesh)
    voxel_grid = grid.from_pv_voxels(pv_voxels, RESOLUTION)

    LegoBrickGrid = []
    it = np.log2(CONST.BIGGEST_BRICK_RES)
    while it >= 0:
        res = 2 ** it
        LegoBrickGrid.append([[[None for _ in range(int(target_res/res))] 
                              for _ in range(int(target_res/res))] 
                              for _ in range(int(target_res/res))])
        it -= 1

    for i in range(target_res):
        for j in range(target_res):
            for k in range(target_res):
                voxel_subgrid = voxel_grid[i*CONST.GROUP_RES : (i+1)*CONST.GROUP_RES, 
                                           j*CONST.GROUP_RES : (j+1)*CONST.GROUP_RES, 
                                           k*CONST.GROUP_RES : (k+1)*CONST.GROUP_RES]
                check_subspace(voxel_subgrid=voxel_subgrid, 
                               position=(i, j, k), 
                               shape=(1, 1, 1), 
                               LegoBrickGrid=LegoBrickGrid)

    return LegoBrickGrid