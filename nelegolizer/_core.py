import pyvista as pv
import nelegolizer.voxel as vox
import nelegolizer.constants as CONST
import nelegolizer.nn.common as cc
import nelegolizer.nn.nn111 as nn111
import numpy as np
from nelegolizer.data._LegoBrickList import LegoBrickList

def check_subspace(grid, pos, shape, dynamic_grid):
    x, y, z = pos

    if shape == (1, 1, 1):
        dynamic_grid[x][y][z][0] = cc.get_brick(nn111.model, grid, CONST.GROUP_RES, pos)


def legolize(path, target_res):
    res = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    voxels = vox.voxelize.from_mesh(mesh, res, 1)
    raw_grid = vox.into_grid(voxels.cell_centers().points, res)

    dynamic_grid = np.zeros([target_res, target_res, target_res, 1], dtype=LegoBrickList)
    for i in range(target_res):
        for j in range(target_res):
            for k in range(target_res):
                subgrid = raw_grid[i*CONST.GROUP_RES : (i+1)*CONST.GROUP_RES, 
                                j*CONST.GROUP_RES : (j+1)*CONST.GROUP_RES, 
                                k*CONST.GROUP_RES : (k+1)*CONST.GROUP_RES]
                check_subspace(subgrid, (i, j, k), (1, 1, 1), dynamic_grid)

    return dynamic_grid