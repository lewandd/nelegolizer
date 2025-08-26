"""
Load models from models/ to python dictionary 'models'
"""

from importlib.machinery import SourceFileLoader
import importlib.util
from nelegolizer import path
import pandas as pd
from typing import Tuple

bc_models_loader = SourceFileLoader("bc_models", path.BRICK_MODULES_FILE)
bc_models_spec = importlib.util.spec_from_loader(name=bc_models_loader.name,
                                                 loader=bc_models_loader)
bc_models_module = importlib.util.module_from_spec(bc_models_spec)
bc_models_loader.create_module(bc_models_spec)
bc_models_loader.exec_module(bc_models_module)
shape_model_map = bc_models_module.load_shape_model_map()

_DIVISIONS_DF = pd.read_csv(path.DIVISIONS_CSV).set_index("id")
division_by_id = {}
for id in _DIVISIONS_DF.index.tolist():
    shape = tuple(map(int, _DIVISIONS_DF.loc[id]["shape"].split(",")))
    offset1 = tuple(map(int, _DIVISIONS_DF.loc[id]["offset1"].split(",")))
    shape1 = tuple(map(int, _DIVISIONS_DF.loc[id]["shape1"].split(",")))
    offset2 = tuple(map(int, _DIVISIONS_DF.loc[id]["offset2"].split(",")))
    shape2 = tuple(map(int, _DIVISIONS_DF.loc[id]["shape2"].split(",")))
    division_by_id[id] = (shape, offset1, shape1, offset2, shape2)

_LABELS_DF = pd.read_csv(path.LABELS_CSV).set_index("id")
shape_label_part_id_map = {}
shape_label_division_id_map = {}
for id in _LABELS_DF.index.tolist():
    shape = tuple(map(int, _LABELS_DF.loc[id]["shape"].split(",")))
    label = _LABELS_DF.loc[id]["label"]
    type_ = _LABELS_DF.loc[id]["type"]
    type_id = _LABELS_DF.loc[id]["type_id"]
    
    if type_ == "brick":
        if not shape in shape_label_part_id_map:
            shape_label_part_id_map[shape] = {}
        shape_label_part_id_map[shape][label] = type_id
    elif type_ == "division":
        if not shape in shape_label_division_id_map:
            shape_label_division_id_map[shape] = {}
        shape_label_division_id_map[shape][label] = type_id

def rotate_division(division: Tuple[int], rotation: int) -> Tuple[int]:
    #TODO needs testing
    if rotation == 0:
        return division
    else:
        shape, offset1, shape1, offset2, shape2 = division
        sh_x, sh_y, sh_z = shape
        off1_x, off1_y, off1_z = offset1
        sh1_x, sh1_y, sh1_z = shape1
        off2_x, off2_y, off2_z = offset2
        sh2_x, sh2_y, sh2_z = shape2
        if rotation == -90:
            return ((sh_z, sh_y, sh_x), 
                    (sh_z-off1_z, off1_y, off1_x),
                    (sh1_z, sh1_y, sh1_x),
                    (sh_z-off2_z, off2_y, off2_x),
                    (sh2_z, sh2_y, sh2_x))
        if rotation == -180:
            return ((sh_x, sh_y, sh_z), 
                    (sh_x-off1_x, off1_y, sh_z-off1_z),
                    (sh1_x, sh1_y, sh1_z),
                    (sh_x-off2_x, off2_y, sh_x-off2_z),
                    (sh2_x, sh2_y, sh2_z))
        if rotation == -270:
            return ((sh_z, sh_y, sh_x), 
                    (off1_z, off1_y, sh_x-off1_x),
                    (sh1_z, sh1_y, sh1_x),
                    (off2_z, off2_y, sh_x-off2_x),
                    (sh2_z, sh2_y, sh2_x))

