import pyvista as pv
import numpy as np

def find_best_rotation(voxel_grid: np.ndarray) -> int:
  shape = voxel_grid.shape
  rotation_score = {"0": 0, "90": 0, "180": 0, "270": 0}

  for x in range(shape[0]):
     for y in range(shape[1]):
        for z in range(shape[2]):
           if voxel_grid[x][y][z]:
            x_inverted = (shape[0]-1) - x
            z_inverted = (shape[2]-1) - z
            rotation_score["0"] += x_inverted + z_inverted
            rotation_score["90"] += x + z_inverted
            rotation_score["180"] += x + z
            rotation_score["270"] += x_inverted + z

  best_rotation = max(rotation_score, key=rotation_score.get)
  return int(best_rotation)

def rotate(grid: np.ndarray, rotation: int) -> np.ndarray:
  shape = grid.shape
  match rotation:
     case 0:
       rotated_grid = grid
     case 90:
       rot_shape = (shape[2], shape[1], shape[0])
       rotated_grid = [[[grid[k][j][i] for k in reversed(range(rot_shape[2]))] 
                                          for j in range(rot_shape[1])] 
                                          for i in range(rot_shape[0])]
     case 180:
       rot_shape = shape
       rotated_grid = [[[grid[i][j][k] for k in reversed(range(rot_shape[2]))] 
                                          for j in range(rot_shape[1])] 
                                          for i in reversed(range(rot_shape[0]))]
     case 270:
       rot_shape = (shape[2], shape[1], shape[0])
       rotated_grid = [[[grid[k][j][i] for k in range(rot_shape[2])] 
                                          for j in range(rot_shape[1])] 
                                          for i in reversed(range(rot_shape[0]))]
     case _:
      raise Exception(f"Rotation can be either 0, 90, 180 or 270. Got {rotation}.")
  return np.array(rotated_grid)

def get_fill_ratio(grid: np.ndarray) -> float:
  fill = 0
  shape = grid.shape
  for i in range(shape[0]):
     for j in range(shape[1]):
        for k in range(shape[2]):
           if (grid[i][j][k]):
              fill += 1
  return fill/(shape[0]*shape[1]*shape[2])

def from_pv_voxels(pv_voxels: pv.UnstructuredGrid, res: int) -> np.ndarray:
    voxel_centers = pv_voxels.cell_centers().points
    grid = np.zeros([res,res,res], dtype=bool)

    for v in voxel_centers:
        vx, vy, vz = v
        grid[int(vx)][int(vy)][int(vz)] = True
    return grid
