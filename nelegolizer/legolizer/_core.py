import pyvista as pv
import nelegolizer.legolizer.voxel as vox
import nelegolizer.constants as CONST
import numpy as np
from nelegolizer.data import LegoBrick, LegoBrickList
import nelegolizer.model.object as obj
import nelegolizer

fill_treshold = 0.1

def find_best_rotation(group, gres):
  """Find rotation degrees (0, 90, 180 or 270) to minimalize calculated value

  Calculated value is sum of X + Z axis

  Args:
    group (list) : list of bools with shape (gres, gres, gres)
    gres (int) : used to determine shape

  Returns:
    int : found rotation (0, 90, 180 or 270)
  """ 
  
  # calculate value for every rotation
  rotations = [
    [0,   sum(x + z                   if group[x][y][z] else 0 for x in range(gres) for y in range(gres) for z in range(gres))],
    [90,  sum((gres-1)-x + z          if group[x][y][z] else 0 for x in range(gres) for y in range(gres) for z in range(gres))],
    [180, sum((gres-1)-x + (gres-1)-z if group[x][y][z] else 0 for x in range(gres) for y in range(gres) for z in range(gres))],
    [270, sum(x + (gres-1)-z          if group[x][y][z] else 0 for x in range(gres) for y in range(gres) for z in range(gres))],
  ]
  best_rotation = min(rotations, key=lambda x:x[1])[0]
  return best_rotation

def rotate_group(group, gres, rotation):
  """Rotate voxel group 0, 90, 180 or 270 degrees by y axis

  Args:
    group (list) : list of bools with shape (gres, gres, gres)
    gres (int) : used to determine shape
    rotation (int) : rotation degrees (0, 90, 180 or 270)

  Returns:
    list : rotated group (with the same shape)
  """ 
  rotated_group = group
  match rotation:
     case 90:
        rotated_group = group[::-1, :, :]
     case 180:
        rotated_group = group[::-1, :, ::-1]
     case 270:
        rotated_group = group[:, :, ::-1]
  return rotated_group

def get_brick(model, group, gres, position):
  """Choose brick most matching the given voxel group

  Args:
    model (NeuralNetwork) : neural network model
    group (list) : list of bools with shape (gres, gres, gres)
    gres (int) : used to determine shape
    position (list) : global positon of group/brick 

  Returns:
    LegoBrickList : list of LegoBrick containing single chosen brick
  """
  best_rotation = find_best_rotation(group, gres)
  group = rotate_group(group, gres, best_rotation)

  fill = 0
  for i in range(gres):
     for j in range(gres):
        for k in range(gres):
           if (group[i][j][k]):
              fill += 1
  #fill = list(filter(lambda a: a != 0, group))
  count = fill/(gres*gres*gres)
  if count > fill_treshold:
    label = obj.test_predict(model, group.flatten())
    lego_brick = LegoBrick(label, position, best_rotation)
    #333print(lego_brick)
    return LegoBrickList([lego_brick])
  else:
     return None
  
def check_subspace(grid, pos, shape, dynamic_grid):
    x, y, z = pos

    if shape == (1, 1, 1):
        dynamic_grid[0][x][y][z] = get_brick(nelegolizer.model.models["model_n111"], grid, CONST.GROUP_RES, pos)

def legolize(path, target_res):
    res = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    voxels = vox.voxelize_from_mesh(mesh, res, 1)
    raw_grid = vox.into_grid(voxels.cell_centers().points, res)

    dynamic_grid = []
    it = np.log2(CONST.BIGGEST_BRICK_RES)
    while it >= 0:
        res = 2 ** it
        dynamic_grid.append(np.zeros([int(target_res/res), int(target_res/res), int(target_res/res)], dtype=LegoBrickList))
        it -= 1

    for i in range(target_res):
        for j in range(target_res):
            for k in range(target_res):
                subgrid = raw_grid[i*CONST.GROUP_RES : (i+1)*CONST.GROUP_RES, 
                                j*CONST.GROUP_RES : (j+1)*CONST.GROUP_RES, 
                                k*CONST.GROUP_RES : (k+1)*CONST.GROUP_RES]
                check_subspace(subgrid, (i, j, k), (1, 1, 1), dynamic_grid)

    return dynamic_grid