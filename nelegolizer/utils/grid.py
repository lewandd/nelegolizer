import pyvista as pv
import numpy as np

def get_grid_from_voxels(voxel_centers, res):
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

def group_grid(grid, gres):
    """Group whole voxel grid into groups of voxels

    Args:
    	grid (list): list of bools with shape (grid_res, grid_res, grid_res)
    	gres (int): target resolution of single voxel group

    Returns:
    	list: list of groups with shape (n_groups, n_groups, n_groups, gres, gres, gres), where n_groups is number of groups in one dimension
    """

    # calculate number of groups in one dimension (expand grid if necessary)
    res = grid.shape[0]
    ext = 0 if res%gres == 0 else gres - res%gres
    n_groups = int((res + ext) / gres)

    # for each group assign correct voxels from grid
    groups = np.zeros((n_groups, n_groups, n_groups, gres, gres, gres), dtype=bool)
    for gi in range(n_groups):
        for gj in range(n_groups):
            for gk in range(n_groups):
                for vi in range(gres):
                    for vj in range(gres):
                        for vk in range(gres):
                            groups[gi][gj][gk][vi][vj][vk] = grid[gi*gres+vi][gj*gres+vj][gk*gres+vk]
    return groups

