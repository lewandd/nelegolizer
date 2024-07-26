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

def rotate_group(group: list[list[list[int]]], gres, rotation: int) -> list[list[list[int]]]:
  shape = get_group_shape(group)
  match rotation:
     case 0:
       rotated_group = group
     case 90:
       rot_shape = (shape[2], shape[1], shape[0])
       rotated_group = [[[group[k][j][i] for k in reversed(range(rot_shape[2]))] 
                                          for j in range(rot_shape[1])] 
                                          for i in range(rot_shape[0])]
     case 180:
       rot_shape = shape
       rotated_group = [[[group[i][j][k] for k in reversed(range(rot_shape[2]))] 
                                          for j in range(rot_shape[1])] 
                                          for i in reversed(range(rot_shape[0]))]
     case 270:
       rot_shape = (shape[2], shape[1], shape[0])
       rotated_group = [[[group[k][j][i] for k in range(rot_shape[2])] 
                                          for j in range(rot_shape[1])] 
                                          for i in reversed(range(rot_shape[0]))]
     case _:
      raise Exception(f"Rotation can be either 0, 90, 180 or 270. Got {rotation}.")
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
