import pyvista as pv
import numpy as np
from pyvista import CellType
import nelegolizer.constants as CONST 

def from_mesh(mesh: pv.PolyData, 
              res: int, 
              dens: float) -> pv.UnstructuredGrid:
    mesh = __scale_to_fill_bound(mesh, [res, res, res])

    # define mesh bounds and lengths
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh_xlen, mesh_ylen, mesh_zlen = xmax-xmin, ymax-ymin, zmax-zmin
    
    # scale mesh by epsilon factor to propertly include border cells
    eps = dens/2
    ext_mesh_xlen, ext_mesh_ylen, ext_mesh_zlen = mesh_xlen+eps, mesh_ylen+eps, mesh_zlen+eps
    ext_scale_ratio = (ext_mesh_xlen/mesh_xlen, ext_mesh_ylen/mesh_ylen, ext_mesh_zlen/mesh_zlen)
    ext_mesh = mesh.scale(ext_scale_ratio)        
    
    # translate to start at position (0, 0, 0)
    ext_mesh.translate((-xmin, -ymin, -zmin), inplace=True)
    ext_mesh.translate((eps/2, eps/2, eps/2), inplace=True)

    return pv.voxelize(ext_mesh, density=dens, check_surface=False)

def from_grid(grid: list[list[list[bool]]], 
              res: int) -> pv.UnstructuredGrid:
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
    cell_points = cell_points*CONST.GROUP_RES

    cpoints = []
    for i in range(len(cells)):
        cpoints.append(8)
        for j in range(8):
            cpoints.append(i*8 + j)    

    cell_type = np.array([CellType.VOXEL for _ in range(len(cells))])
    return pv.UnstructuredGrid(np.array(cpoints), cell_type, np.array(cell_points).astype(float))

def __scale_to_fill_bound(mesh: pv.PolyData, 
                          bound: tuple[int, int, int]) -> pv.PolyData:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    xlen, ylen, zlen = xmax-xmin, ymax-ymin, zmax-zmin
    max_len = max([xlen, ylen, zlen])
    xres, yres, zres = bound
    return mesh.scale([xres/max_len, yres/max_len, zres/max_len])
