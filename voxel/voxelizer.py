import pyvista as pv
import numpy as np
from pyvista import CellType

def voxelize_from_mesh(mesh, res, dens):
    """Turns the mesh into a voxel dataset

    Args:
        mesh (pyvista.PolyData) : mesh to voxelize
        res (int) : target cell grid resolution
        dens (float) : density of grid

    Returns:
        pyvista.UnstructuredGrid : dataset of cells defining voxels 

    """
    mesh = __fill_bound(mesh, [res, res, res])

    # define mesh bounds and lengths
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    xlen, ylen, zlen = xmax-xmin, ymax-ymin, zmax-zmin
    
    # scale mesh by epsilon factor to propertly include border cells
    eps = dens/2
    xres, yres, zres = [xlen+eps, ylen+eps, zlen+eps]
    ex_mesh = mesh.scale([xres/xlen, yres/ylen, zres/zlen])        
    
    # translate to start at position (0, 0, 0)
    ex_mesh.translate((-xmin, -ymin, -zmin), inplace=True)
    ex_mesh.translate((eps/2, eps/2, eps/2), inplace=True)

    return pv.voxelize(ex_mesh, density=dens)

def voxelize_from_grid(grid, res):
    """Turns the grid into a voxel dataset

    Args:
        grid (list) : grid to voxelize, (res, res, res) shape list of bools
        res (int) : resolution of grid array

    Returns:
        pyvista.UnstructuredGrid : dataset of cells defining voxels 

    """
    cells = []
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if grid[i][j][k]:
                    cell = [ 
                        [i,    j,    k],
                        [i+1,  j,    k],
                        [i,  j+1,  k],
                        [i+1  ,  j+1,  k], 
                        [i,    j,    k+1],
                        [i+1,  j,    k+1],
                        [i,  j+1,  k+1],
                        [i+1,    j+1,  k+1]
                    ]
                    cells.append(cell)
    cell_points = np.vstack(cells)
    cell_points = cell_points*group_res

    cpoints = []
    for i in range(len(cells)):
        cpoints.append(8)
        for j in range(8):
            cpoints.append(i*8 + j)    

    cell_type = np.array([CellType.VOXEL for _ in range(len(cells))])
    return pv.UnstructuredGrid(np.array(cpoints), cell_type, np.array(cell_points).astype(float))

def __fill_bound(mesh, bound):
    """Scale mesh by the same factor for all dimensions to maximally fill a bound 

    Mesh proportions don't change. Scale factor for X, Y, Z dimensions are the same.

    Args:
        mesh (pyvista.PolyData) : mesh to scale
        bound (list) : boundary for mesh, (3) shape list of ints
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    xlen, ylen, zlen = xmax-xmin, ymax-ymin, zmax-zmin
    max_len = max([xlen, ylen, zlen])
    xres, yres, zres = bound
    return mesh.scale([xres/max_len, yres/max_len, zres/max_len])

