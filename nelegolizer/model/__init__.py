"""
Load models from models/ to python dictionary 'models'
"""

from importlib.machinery import SourceFileLoader
import importlib.util
from nelegolizer import path
import pandas as pd
from typing import Tuple
from nelegolizer.model.io import load_model_str

shape_model_map = None
division_by_id = None
shape_label_part_id_map = None
shape_label_division_id_map = None
subshapes_by_id = {}
shape_label_part_id_map2 = {}
shape_label_part_rot_map2 = {}
shape_label_subshape_id_map2 = {}

shape_model_map_cnn = {}

def initilize_models_cnn():
    global shape_model_map_cnn
    shape_model_map_cnn[(2,3,2)] = load_model_str(path.MODEL_232, (10,6,10))
    shape_model_map_cnn[(2,3,1)] = load_model_str(path.MODEL_231, (10,6,5))
    shape_model_map_cnn[(1,3,1)] = load_model_str(path.MODEL_131, (5,6,5))
    shape_model_map_cnn[(1,1,1)] = load_model_str(path.MODEL_111, (5,2,5))

def initilize_models_csv():
    global subshapes_by_id
    global shape_label_part_id_map2
    global shape_label_part_rot_map2
    global shape_label_subshape_id_map2

    _SUBSHAPES_DF = pd.read_csv(path.SUBSHAPES_CSV).set_index("id")
    for id in _SUBSHAPES_DF.index.tolist():
        shape = tuple(map(int, _SUBSHAPES_DF.loc[id]["shape"].split(",")))
        sub_shape = tuple(map(int, _SUBSHAPES_DF.loc[id]["subshape"].split(",")))
        subshapes_by_id[id] = (shape, sub_shape)

    _LABELS2_DF = pd.read_csv(path.LABELS2_CSV).set_index("id")
    for id in _LABELS2_DF.index.tolist():
        shape = tuple(map(int, _LABELS2_DF.loc[id]["shape"].split(",")))
        label = _LABELS2_DF.loc[id]["label"]
        type_ = _LABELS2_DF.loc[id]["type"]
        type_id = _LABELS2_DF.loc[id]["type_id"]
        rotation=_LABELS2_DF.loc[id]["rotation"]
        
        if type_ == "brick":
            if not shape in shape_label_part_id_map2:
                shape_label_part_id_map2[shape] = {}
            if not shape in shape_label_part_rot_map2:
                shape_label_part_rot_map2[shape] = {}
            shape_label_part_id_map2[shape][label] = type_id
            shape_label_part_rot_map2[shape][label] = rotation
        elif type_ == "subshape":
            if not shape in shape_label_subshape_id_map2:
                shape_label_subshape_id_map2[shape] = {}
            shape_label_subshape_id_map2[shape][label] = type_id

def initilize_models():
    global shape_model_map
    global division_by_id
    global shape_label_part_id_map
    global shape_label_division_id_map
    global subshapes_by_id
    global shape_label_part_id_map2
    global shape_label_part_rot_map2
    global shape_label_subshape_id_map2
    
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

    _SUBSHAPES_DF = pd.read_csv(path.SUBSHAPES_CSV).set_index("id")
    subshapes_by_id = {}
    for id in _SUBSHAPES_DF.index.tolist():
        shape = tuple(map(int, _SUBSHAPES_DF.loc[id]["shape"].split(",")))
        sub_shape = tuple(map(int, _SUBSHAPES_DF.loc[id]["subshape"].split(",")))
        subshapes_by_id[id] = (shape, sub_shape)

    _LABELS2_DF = pd.read_csv(path.LABELS2_CSV).set_index("id")
    shape_label_part_id_map2 = {}
    shape_label_part_rot_map2 = {}
    shape_label_subshape_id_map2 = {}
    for id in _LABELS2_DF.index.tolist():
        shape = tuple(map(int, _LABELS2_DF.loc[id]["shape"].split(",")))
        label = _LABELS2_DF.loc[id]["label"]
        type_ = _LABELS2_DF.loc[id]["type"]
        type_id = _LABELS2_DF.loc[id]["type_id"]
        rotation=_LABELS2_DF.loc[id]["rotation"]
        
        if type_ == "brick":
            if not shape in shape_label_part_id_map2:
                shape_label_part_id_map2[shape] = {}
            if not shape in shape_label_part_rot_map2:
                shape_label_part_rot_map2[shape] = {}
            shape_label_part_id_map2[shape][label] = type_id
            shape_label_part_rot_map2[shape][label] = rotation
        elif type_ == "subshape":
            if not shape in shape_label_subshape_id_map2:
                shape_label_subshape_id_map2[shape] = {}
            shape_label_subshape_id_map2[shape][label] = type_id

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

