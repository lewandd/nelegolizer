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

def get_group_fill_ratio(group: list[list[list[int]]]) -> float:
  fill = 0
  shape = (len(group), len(group[0]), len(group[0][0]))
  for i in range(shape[0]):
     for j in range(shape[1]):
        for k in range(shape[2]):
           if (group[i][j][k]):
              fill += 1
  return fill/(shape[0]*shape[1]*shape[2])
