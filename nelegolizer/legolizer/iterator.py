import numpy as np
from nelegolizer.data import LegoBrick
from nelegolizer.utils.conversion import *

label_data = {3: {1: ("3004", 0),
                  2: ("3004", 90),
                  3: ("3004", 180),
                  4: ("3004", 270),
                  5: ("3005", 0),
                  6: ("54200", 0),
                  7: ("54200", 90),
                  8: ("54200", 180),
                  9: ("54200", 270),},
              1: {1: ("3024", 0),}}

def find_next_to_cover_net(gc, bc, analyzed):
    geometry = gc.brick_grid
    studs = bc.top_available_grid
    tubes = bc.bottom_available_grid
    tubes3 = bc.bottom3_available_grid
    analyzed1 = analyzed[1]
    analyzed3 = analyzed[3]
    
    def get_sorted_indices(mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        sorted_indices = indices[np.lexsort((indices[:, 0], indices[:, 2], -indices[:, 1]))]
        return sorted_indices[0]

    all_connections_pos1 = np.logical_or(studs, tubes)
    all_connections_pos3 = np.logical_or(studs, tubes3)

    to_cover1 = np.logical_and(np.logical_and(all_connections_pos1, geometry), ~analyzed1)
    to_cover3 = np.logical_and(np.logical_and(all_connections_pos3, geometry), ~analyzed3)

    return get_sorted_indices(to_cover1), get_sorted_indices(to_cover3)

def place_brick(label, pos, net_type, bc):
    id, rot = label_data[net_type][label]
    new_pos = pos
    if id == "3004" and rot == 180:
        new_pos = pos - np.array([1, 0, 0])
    if id == "3004" and rot == 270:
        new_pos = pos - np.array([0, 0, 1])  
    new_brick = LegoBrick(id=id, 
                          mesh_position=bu_to_mesh(new_pos), 
                          rotation=rot)
            
    if bc.is_placement_available(new_brick):
        bc.place_brick(new_brick)
        return True
    else:
        return False