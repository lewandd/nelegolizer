import pyvista as pv
import nelegolizer.constants as CONST
import numpy as np
from nelegolizer.data import LegoBrick
import nelegolizer.model.object as obj
from torch import nn
import nelegolizer
from nelegolizer.utils.group import find_best_rotation, rotate_group, get_group_fill_ratio
from nelegolizer.utils.voxelization import voxelize_from_mesh, into_grid

fill_treshold = 0.1

def get_brick(model: nn.Module, 
              group: list[list[list[int]]], 
              gr_res: int, 
              gr_pos: tuple[int, int, int]) -> LegoBrick:
  best_rotation = find_best_rotation(group)
  group = rotate_group(group, best_rotation)

  fill_ratio = get_group_fill_ratio(group)
  if fill_ratio > fill_treshold:
    label = obj.test_predict(model, group.flatten())
    lego_brick = LegoBrick(label=label, position=gr_pos, rotation=best_rotation)
    return lego_brick
  else:
     return None
  
def check_subspace(grid, pos, shape, dynamic_grid):
    x, y, z = pos

    if shape == (1, 1, 1):
        dynamic_grid[0][x][y][z] = get_brick(nelegolizer.model.models["model_n111"], grid, CONST.GROUP_RES, pos)

def legolize(path, target_res):
    res = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    voxels = voxelize_from_mesh(mesh, res, 1)
    raw_grid = into_grid(voxels.cell_centers().points, res)

    dynamic_grid = []
    it = np.log2(CONST.BIGGEST_BRICK_RES)
    while it >= 0:
        res = 2 ** it
        dynamic_grid.append([[[None for _ in range(int(target_res/res))] 
                              for _ in range(int(target_res/res))] 
                              for _ in range(int(target_res/res))])
        it -= 1

    for i in range(target_res):
        for j in range(target_res):
            for k in range(target_res):
                subgrid = raw_grid[i*CONST.GROUP_RES : (i+1)*CONST.GROUP_RES, 
                                j*CONST.GROUP_RES : (j+1)*CONST.GROUP_RES, 
                                k*CONST.GROUP_RES : (k+1)*CONST.GROUP_RES]
                check_subspace(subgrid, (i, j, k), (1, 1, 1), dynamic_grid)

    return dynamic_grid