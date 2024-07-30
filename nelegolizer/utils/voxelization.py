import pyvista as pv
import numpy as np
from pyvista import CellType

import nelegolizer.constants as CONST 
from nelegolizer.utils import mesh as umesh

def from_mesh(mesh: pv.PolyData, 
              res: int, 
              dens: float) -> pv.UnstructuredGrid:
    mesh = umesh.scale_to(mesh, (res, res, res), keep_ratio=True)

    # extend mesh by epsilon for proper voxelization by pyvista.voxelize
    mesh_res = umesh.get_resolution(mesh)
    eps = dens/2     
    eps_ext_mesh = umesh.scale_to(mesh, mesh_res+eps, keep_ratio=False)
    
    eps_ext_mesh = umesh.translate_to_zero(eps_ext_mesh)
    return pv.voxelize(eps_ext_mesh, density=dens, check_surface=False)

def from_grid(grid: np.ndarray, 
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