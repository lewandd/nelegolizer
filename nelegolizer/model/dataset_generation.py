from torch import nn
import os
import torch
import pyvista as pv
import numpy as np
from typing import Tuple, List
from nelegolizer.data import LDrawFile, initilize_parts, BrickOccupancy, ObjectOccupancy
from nelegolizer import const
from nelegolizer.utils.grid import bu_to_vu, vu_to_bu
import nelegolizer.utils.voxelization as uvox
import json
from nelegolizer.data import ClassificationResult2
from nelegolizer.model import initilize_models_csv, subshapes_by_id, shape_label_part_id_map2,shape_label_part_rot_map2,shape_label_subshape_id_map2

def sample_to_str(channel1: np.ndarray, channel2: np.ndarray, label: int) -> str:
    """
    Convert two 3D bool ndarrays and single int label into one-line str (JSON)
    """
    assert channel1.shape == channel2.shape, "Obie siatki muszą mieć ten sam wymiar"
    assert channel1.dtype == bool and channel2.dtype == bool, "Siatki powinny być typu bool"

    data_dict = {
        "channel1": channel1.astype(int).flatten().tolist(),
        "channel2": channel2.astype(int).flatten().tolist(),
        "shape": channel1.shape,
        "label": int(label)
    }
    return json.dumps(data_dict)

def save_dataset(samples: list, filename: str):
    """
    Save samples to txt file.
    """
    with open(filename, "w") as f:
        for s in samples:
            f.write(s + "\n")

shape_id_rot_label_map = {(2, 3, 2): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:1, 90:2, 180:1, 270:2}},
                          (2, 3, 1): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:2, 90:1, 180:2, 270:1}},
                          (1, 3, 1): {"3005": {0:2, 90:2, 180:2, 270:2},
                                      "54200": {0:3, 90:4, 180:5, 270:6},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:2, 90:2, 180:2, 270:2}},
                          (1, 1, 1): {"3005": {0:1, 90:1, 180:1, 270:1},
                                      "54200": {0:1, 90:1, 180:1, 270:1},
                                      "3024": {0:1, 90:1, 180:1, 270:1},
                                      "3004": {0:1, 90:1, 180:1, 270:1}}}

# TODO jeszcze trzeba uwzględnić obrót odpowiednio

# TODO poza tym mogę się czatu GPT spytać co zrobić z tymi pustmi miejscami czy
# nie lepiej byłoby je pomijać skoro wiele danych jest takich samych?
# jak to wpływa na sieć? czy to jest potrzebne czy nie?

def make_samples(oo: ObjectOccupancy, shape: Tuple, bricks: list, debug = False) -> List[str]:
    initilize_models_csv()

    samples = []

    if debug:
        print("grid shape:", oo.brick_grid.shape)
    
    # TODO możnaby ładniej iterować po tym wszystkim
    for y in range(oo.brick_grid.shape[1] - shape[1], -1, -1):
        for x in range(oo.brick_grid.shape[0] - shape[0] + 1):
            for z in range(oo.brick_grid.shape[2] - shape[2] + 1):
                pos = np.array([x, y, z])
                vu_pos = bu_to_vu(pos)
                vu_shape = bu_to_vu(shape) + 2*const.PADDING
                #print("voxel group shape:", vu_shape)
                patch = oo.brick_grid[x:x+shape[0], y:y+shape[1], z:z+shape[2]]
                
                channel1 = oo.voxel_grid[vu_pos[0]:vu_pos[0]+vu_shape[0],
                                        vu_pos[1]:vu_pos[1]+vu_shape[1],
                                        vu_pos[2]:vu_pos[2]+vu_shape[2]]

                temp_bo = BrickOccupancy.from_bricks(bricks)
                brick = temp_bo.get_brick_at(bricks, (pos[0], pos[1]+shape[1]-1, pos[2]))

                label = 0 # if nothing to analyse then label is 0 for every cnn

                if brick is not None:
                    label = shape_id_rot_label_map[shape][brick.part.id][brick.rotation]
                    temp_bo.remove_brick(brick)


                # TODO lista 3-wymiarowych tablic np.ndarray określających jakie pozycje
                # trzeba usunąć z analizowanej przestrzeni, przede wszystkim powinna tam
                # tablica z usuniętą pozycją (0, 0, 0)
                # też to bedzie tak na prawdę wielki słownik, gdzie dla każdego kształtu
                # będzie oddzielna lista
                #set_of_possibilities = []

                #deleted_list = []

                #for s in set_of_possibilities:
                #    for (x, y, z), val in np.ndenumerate(s):
                #        # TODO set zamiast list i dodać do LegoBrick __hash__ i __eq__
                #        s_deleted_list = [] 
                #        if val: # brick to delete in this pre-set
                #            brick = temp_bo.get_brick_at(bricks, (pos[0]-+x, pos[1]+shape[1]-1, pos[2]-+z))
                #            if brick is not None:
                #                s_deleted_list.append(brick)
                #        # TODO zadbać o to by klocki w liście się nie powtarzały
                #        if s_deleted_list:
                #            deleted_list.append(s_deleted_list)

                # TODO zadabć by listy w deleted_list się nie powtarzały
                
                #for s_deleted_list in deleted_list:
                #    temp_bo = BrickOccupancy.from_bricks(bricks)
                #    for b in s_deleted_list:
                #        temp_bo.remove_brick

                channel2 = temp_bo.voxel_grid[vu_pos[0]:vu_pos[0]+vu_shape[0],
                                            vu_pos[1]:vu_pos[1]+vu_shape[1],
                                            vu_pos[2]:vu_pos[2]+vu_shape[2]]

                #print("channel1 shape:", channel1.shape)
                #print("channel2 shape:", channel2.shape)
                samples.append(sample_to_str(channel1, channel2, label))
    if debug:
        print(f"samples generated:", len(samples))
    return samples
