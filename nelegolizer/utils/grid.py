import pyvista as pv
import numpy as np

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

def from_pv_voxels(pv_voxels: pv.UnstructuredGrid, res: int) -> np.ndarray:
    voxel_centers = pv_voxels.cell_centers().points
    grid = np.zeros([res,res,res], dtype=bool)
    for vx, vy, vz in voxel_centers:
        grid[int(vx)][int(vy)][int(vz)] = True
    return grid
