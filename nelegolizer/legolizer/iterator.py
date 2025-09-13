import numpy as np
from ..data import LegoBrick, part_by_id
from ..utils.conversion import bu_to_mesh

def find_next_pos_to_cover(gc, bc, analyzed, height_map):
    geometry = gc.brick_grid
    studs = bc.top_available_grid
    tubes = bc.bottom_available_grid
    tubes3 = bc.bottom3_available_grid
    
    def get_sorted_indices(mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        sorted_indices = indices[np.lexsort((indices[:, 0], indices[:, 2], -indices[:, 1]))]
        return sorted_indices[0]

    all_connections_pos = {1: np.logical_or(studs, tubes),
                           3: np.logical_or(studs, tubes3)}

    to_cover = {}
    for subset, a in analyzed.items():
        h = height_map[subset]
        indices = np.logical_and(np.logical_and(all_connections_pos[h], geometry), ~a)
        to_cover[subset] = get_sorted_indices(indices)
    return to_cover

def place_brick(brick_id, rotation, pos, bc):
    new_pos = pos
    if brick_id == "3004" and rotation == 180:
        new_pos = pos - np.array([1, 0, 0])
    if brick_id == "3004" and rotation == 270:
        new_pos = pos - np.array([0, 0, 1])  
    new_brick = LegoBrick(id=brick_id, 
                          mesh_position=bu_to_mesh(new_pos), 
                          rotation=rotation)
            
    if bc.is_placement_available(new_brick):
        bc.place_brick(new_brick)
        return new_brick
    else:
        return None
    
def make_brick_variants(placement_pos, bricks_pool):
    bricks = []
    for brick_id in bricks_pool:
        for rot in [0, 90, 180, 270]:
            part = part_by_id[brick_id]
            pos = placement_pos.copy()

            # TODO to do zmieny na uniwersalnę metodę
            if part.size[0] > 1:
                if int(rot) == 180:
                    pos = placement_pos - np.array([1, 0, 0])
                elif int(rot) == 270:
                    pos = placement_pos - np.array([0, 0, 1])

            bricks.append(LegoBrick(id=brick_id,
                                    mesh_position=bu_to_mesh(pos),
                                    rotation=int(rot),))
    return bricks