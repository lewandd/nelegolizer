import numpy as np

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