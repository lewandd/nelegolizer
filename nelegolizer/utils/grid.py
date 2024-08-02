import pyvista as pv
import numpy as np
from nelegolizer.utils import mesh as umesh

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

def get_subgrid(grid: np.ndarray, position: tuple[int, int, int], shape: np.ndarray) -> np.ndarray:
    start = position
    end =   position + shape
    return grid[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def rotate(grid: np.ndarray, rotation: int) -> np.ndarray:
  match rotation:
     case 0:
       rotated_grid = grid
     case 90:
       rotated_grid = np.zeros_like(np.transpose(grid))
       for (i, j, k), val in np.ndenumerate(grid[::-1,:,:]):
          rotated_grid[k, j, i] = val
     case 180:
       rotated_grid = np.zeros_like(grid)
       for (i, j, k), val in np.ndenumerate(grid[::-1,:,::-1]):
          rotated_grid[i, j, k] = val
     case 270:
       rotated_grid = np.zeros_like(np.transpose(grid))
       for (i, j, k), val in np.ndenumerate(grid[:,:,::-1]):
          rotated_grid[k, j, i] = val
     case _:
      raise Exception(f"Rotation can be either 0, 90, 180 or 270. Got {rotation}.")
  return rotated_grid

def get_fill_ratio(grid: np.ndarray) -> float:
  fill = 0
  volume = grid.shape[0]*grid.shape[1]*grid.shape[2]
  for x in np.nditer(grid):
      fill += 1 if x else 0
  return fill/volume

def from_pv_voxels(pv_voxels: pv.UnstructuredGrid,
                   *, unit_shape: np.ndarray, 
                      required_dim_divisibility: np.ndarray = np.array([1, 1, 1])) -> np.ndarray:
    mesh_shape = umesh.get_resolution(pv_voxels)
    resolution = (mesh_shape/unit_shape).astype(int)

    remainder = resolution % required_dim_divisibility
    for dim in range(resolution.size):     
      if remainder[dim] != 0:
        extended_resolution = resolution[dim] - remainder[dim] + required_dim_divisibility[dim] 
        resolution[dim] = extended_resolution
       
    voxel_centers = pv_voxels.cell_centers().points
    grid = np.zeros(resolution, dtype=bool)
    for position in voxel_centers:
        x, y, z = (position/unit_shape).astype(int)
        grid[x, y, z] = True
    return grid
