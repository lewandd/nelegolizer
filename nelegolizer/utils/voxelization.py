import pyvista as pv
import numpy as np
from pyvista import CellType

import nelegolizer.constants as CONST 
from nelegolizer.utils import mesh as umesh

def from_mesh(mesh: pv.PolyData,
              *, unit_shape: np.ndarray = np.array([0.1, 0.1, 0.1])) -> pv.UnstructuredGrid:
    # extend mesh by epsilon for proper voxelization by pyvista.voxelize
    eps = unit_shape/2.0
    eps_ext_mesh = umesh.scale_to(mesh, umesh.get_resolution(mesh)+eps)

    eps_ext_mesh = umesh.translate_to_zero(eps_ext_mesh)
    return pv.voxelize(eps_ext_mesh, density=unit_shape, check_surface=False)

def from_grid(grid: np.ndarray, 
              *, unit_shape: np.ndarray) -> pv.UnstructuredGrid:
    points_id = {}
    points = []
    id = 0
    for (x, y, z), val in np.ndenumerate(grid):
        if val:
            binary_combinations_222 = np.ndenumerate(np.zeros([2,2,2]))
            for (b3, b2, b1), _ in binary_combinations_222:
                if f"{x+b1}|{y+b2}|{z+b3}" not in points_id:
                    points_id[f"{x+b1}|{y+b2}|{z+b3}"] = id
                    points.append(np.array([x+b1, y+b2, z+b3]) * unit_shape)
                    id += 1

    cells = []
    for (x, y, z), val in np.ndenumerate(grid):
        if val:
            binary_combinations_222 = np.ndenumerate(np.zeros([2,2,2]))
            cells.append([8] + [points_id[f"{x+b1}|{y+b2}|{z+b3}"] for (b3, b2, b1), _ in binary_combinations_222])

    cell_type = [CellType.VOXEL for _ in range(len(cells))]

    return pv.UnstructuredGrid(np.array(cells).flatten(), 
                               np.array(cell_type), 
                               np.array(points).astype(float))