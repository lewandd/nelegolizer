import numpy as np

def get_scalars(voxels, labels, res):
    """ Return list of scalars in order corresponding to cells order
    
    Args:
        voxels (pyvista.UnstructuredGrid) : grid of cells
        labels (list) : grid of shape (res, res, res) assigning labels for locations
        res (int) : resolution of a grid

    Returns:
        list: list of scalars (floats) in order corresponding to cells order
    """

    scalars = []
    for i in range(voxels.GetNumberOfCells()):
        cell = voxels.GetCell(i)
        x, y, z = (cell.GetBounds()[0]/res, cell.GetBounds()[2]/res, cell.GetBounds()[4]/res)
        label = labels[int(x)][int(y)][int(z)]
        for j in range(8):
            scalars.append(label)
    return scalars

def flatten_dynamic_grid(dynamic_grid, target_res, it):
    """ Map dynamic grid structure to casual 3D grid

    Args:
        dynamic_grid (list) : list of 3D grids
        target_res (int) : target grid resolution
        it (int) : number of iterations, every iteration represents 2 ** it resolution
        
    Retuns:
        list : 3D grid of values representing labels
    """

    target_grid = np.zeros([target_res, target_res, target_res], dtype=float)
    while it >= 0:
        res = 2 ** it
        for i in range(len(dynamic_grid[it])):
            for j in range(len(dynamic_grid[it][i])):
                for k in range(len(dynamic_grid[it][i][j])):
                    if dynamic_grid[it][i][j][k]:
                        for l in range(res):
                            for m in range(res):
                                for n in range(res):
                                    target_grid[i*res+l][j*res+m][k*res+n] = dynamic_grid[it][i][j][k].into_list()[0].get_label()+1
        it -= 1
    return target_grid