import pyvista as pv
import numpy as np

def into_grid(voxel_centers, res):
    """Turn voxel cells into grid
    
    Args:
        voxel_centers (pyvista.PolyData) : voxel cells centers
        res (int) : target grid resolution
    
    Returns:
        list: grid of bools defining presence or absence of voxel cells, list with shape (res, res, res) 
    """
    grid = np.zeros([res,res,res], dtype=bool)

    for v in voxel_centers:
        vx, vy, vz = v
        grid[int(vx)][int(vy)][int(vz)] = True
    return grid
