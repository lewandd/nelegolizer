import pyvista as pv
import numpy as np
from nelegolizer.utils import mesh as umesh
from typing import Tuple


def get_fill(voxel_grid: np.ndarray) -> np.ndarray:
    n = 0
    for x in np.nditer(voxel_grid):
        n += 1 if x else 0
    return n


def get_mass_center(voxel_grid: np.ndarray) -> np.ndarray:
    dist_sum = np.array([0.0, 0.0, 0.0])
    n = np.float64(get_fill(voxel_grid))
    if n == 0:
        return None

    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z]:
                    dist_sum = dist_sum + np.array([x+0.5, y+0.5, z+0.5])
    return dist_sum/n


def find_best_rotation(voxel_grid: np.ndarray) -> int:
    rotation_score = {"0": 0, "90": 0, "180": 0, "270": 0}
    for (x, _, z), val in np.ndenumerate(voxel_grid):
        if val:
            x_inverted = (voxel_grid.shape[0]-1) - x
            z_inverted = (voxel_grid.shape[2]-1) - z
            rotation_score["0"] += x_inverted + z_inverted
            rotation_score["90"] += x + z_inverted
            rotation_score["180"] += x + z
            rotation_score["270"] += x_inverted + z
    return int(max(rotation_score, key=rotation_score.get))


def get_subgrid(grid: np.ndarray,
                position: Tuple[int, int, int],
                shape: np.ndarray) -> np.ndarray:
    start = position
    end = position + shape
    if np.any(end > grid.shape) or np.any(start < np.array([0, 0, 0])):
        raise IndexError(f"Cannot get a subgrid with position={position} and"
                         f" shape={shape} from grid with shape={grid.shape}."
                         " Indexes are out of bond.")
    return grid[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def rotate(grid: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        rotated_grid = grid
    elif degrees == 90:
        rotated_grid = np.zeros_like(np.transpose(grid))
        for (i, j, k), val in np.ndenumerate(grid[::-1, :, :]):
            rotated_grid[k, j, i] = val
    elif degrees == 180:
        rotated_grid = np.zeros_like(grid)
        for (i, j, k), val in np.ndenumerate(grid[::-1, :, ::-1]):
            rotated_grid[i, j, k] = val
    elif degrees == 270:
        rotated_grid = np.zeros_like(np.transpose(grid))
        for (i, j, k), val in np.ndenumerate(grid[:, :, ::-1]):
            rotated_grid[k, j, i] = val
    else:
        raise Exception("Rotation can be either 0, 90, 180 or 270 degrees. " +
                        f"Got {degrees}.")
    return rotated_grid


def get_fill_ratio(grid: np.ndarray) -> float:
    fill = get_fill(grid)
    volume = grid.shape[0]*grid.shape[1]*grid.shape[2]
    return fill/volume


def from_pv_voxels(pv_voxels: pv.UnstructuredGrid) -> np.ndarray:
    pv_voxels = umesh.translate_to_zero(pv_voxels)
    mesh_shape = umesh.get_resolution(pv_voxels)
    unit_shape = umesh.get_resolution(pv_voxels.extract_cells(0))
    resolution = np.ceil((mesh_shape/unit_shape)).astype(int)
    voxel_centers = pv_voxels.cell_centers().points
    grid = np.zeros(resolution, dtype=bool)
    for position in voxel_centers:
        x, y, z = (position/unit_shape).astype(int)
        grid[x, y, z] = True
    return grid


def add_padding(grid: np.ndarray,
                padding: np.ndarray) -> np.ndarray:
    grid_extended = np.zeros(grid.shape + padding * 2, dtype=bool)
    start = padding
    end = start + grid.shape
    grid_extended[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = grid
    return grid_extended


def provide_divisibility(grid: np.ndarray,
                         divider: np.ndarray) -> np.ndarray:
    resolution = np.array(grid.shape)
    remainder = resolution % divider
    for dim in range(resolution.size):
        if remainder[dim] != 0:
            ext_resolution = resolution[dim] - remainder[dim] + divider[dim]
            resolution[dim] = ext_resolution
    extended_grid = np.zeros(resolution, dtype=bool)
    extended_grid[:grid.shape[0], :grid.shape[1], :grid.shape[2]] = grid
    return extended_grid


def from_mesh(mesh: pv.PolyData,
              *, voxel_mesh_shape: np.ndarray) -> np.ndarray:
    # copied voxelization.from_mesh code to avoid import
    eps = voxel_mesh_shape/2.0
    eps_ext_mesh = umesh.scale_to(mesh, umesh.get_resolution(mesh)+eps)

    eps_ext_mesh = umesh.translate_to_zero(eps_ext_mesh)
    pv_voxels = pv.voxelize(eps_ext_mesh,
                            density=voxel_mesh_shape,
                            check_surface=False)
    return from_pv_voxels(pv_voxels)
