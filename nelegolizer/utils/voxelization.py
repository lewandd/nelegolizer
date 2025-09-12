from . import mesh as utils_mesh
from . import grid as utils_grid
from ..constants import VU, BU_RES
import pyvista as pv
import numpy as np
from pyvista import CellType
from typing import Union

def voxelize(mesh: Union[str, pv.PolyData]):
    if isinstance(mesh, str):
        mesh_file_path = mesh
        reader = pv.get_reader(mesh_file_path)
        mesh = reader.read()
    elif not isinstance(mesh, pv.PolyData):
        raise ValueError("Legolize argument shold be either 3d object "
                         "path (str) or mesh (pyvista.PolyData)")
    mesh = mesh.flip_normal([0.0, 1.0, 0.0])

    voxel_grid = utils_grid.from_mesh(mesh, voxel_mesh_shape=VU)
    voxel_grid = utils_grid.provide_divisibility(
                                    voxel_grid,
                                    divider=BU_RES)
    return voxel_grid

def from_mesh(mesh: pv.PolyData,
              *, voxel_mesh_shape: np.ndarray) -> pv.UnstructuredGrid:
    # extend mesh by epsilon for proper voxelization by pyvista.voxelize
    eps = voxel_mesh_shape/2.0
    eps_ext_mesh = utils_mesh.scale_to(mesh, utils_mesh.get_resolution(mesh)+eps)

    eps_ext_mesh = utils_mesh.translate_to_zero(eps_ext_mesh)
    return pv.voxelize(eps_ext_mesh,
                       density=voxel_mesh_shape,
                       check_surface=False)


def from_grid(grid: np.ndarray,
              *, voxel_mesh_shape: np.ndarray) -> pv.UnstructuredGrid:
    points_id = {}
    points = []
    id = 0
    for (x, y, z), val in np.ndenumerate(grid):
        if val:
            bin_combinations_222 = np.ndenumerate(np.zeros([2, 2, 2]))
            for (b3, b2, b1), _ in bin_combinations_222:
                if f"{x+b1}|{y+b2}|{z+b3}" not in points_id:
                    points_id[f"{x+b1}|{y+b2}|{z+b3}"] = id
                    point = np.array([x+b1, y+b2, z+b3])
                    points.append(point * voxel_mesh_shape)
                    id += 1

    cells = []
    for (x, y, z), val in np.ndenumerate(grid):
        if val:
            bin_combinations_222 = np.ndenumerate(np.zeros([2, 2, 2]))
            cells.append([8] + [points_id[f"{x+b1}|{y+b2}|{z+b3}"]
                                for (b3, b2, b1), _ in bin_combinations_222])

    cell_type = [CellType.VOXEL for _ in range(len(cells))]

    return pv.UnstructuredGrid(np.array(cells).flatten(),
                               np.array(cell_type),
                               np.array(points).astype(float))
