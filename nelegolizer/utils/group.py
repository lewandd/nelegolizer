def get_group_shape(group: list[list[list[int]]]) -> tuple[int, int, int]:
   return (len(group), len(group[0]), len(group[0][0]))

def find_best_rotation(group: list[list[list[int]]]) -> int:
  shape = get_group_shape(group)
  rotation_score = {"0": 0, "90": 0, "180": 0, "270": 0}

  for x in range(shape[0]):
     for y in range(shape[1]):
        for z in range(shape[2]):
           if group[x][y][z]:
            x_inverted = (shape[0]-1) - x
            z_inverted = (shape[2]-1) - z
            rotation_score["0"] += x_inverted + z_inverted
            rotation_score["90"] += x + z_inverted
            rotation_score["180"] += x + z
            rotation_score["270"] += x_inverted + z

  best_rotation = max(rotation_score, key=rotation_score.get)
  return int(best_rotation)

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
  shape = get_group_shape(group)
  for i in range(shape[0]):
     for j in range(shape[1]):
        for k in range(shape[2]):
           if (group[i][j][k]):
              fill += 1
  return fill/(shape[0]*shape[1]*shape[2])
