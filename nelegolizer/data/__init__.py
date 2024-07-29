import pandas as pd
import nelegolizer.constants as CONST

from ._LegoBrick import LegoBrick
from ._LDrawPart import LDrawPart

_PART_LABEL_PATH = CONST.PATH + "/LDraw/part/part_label.csv"
_PART_LABEL_DF = pd.read_csv(_PART_LABEL_PATH).set_index("label")

_PART_DETAILS_PATH = CONST.PATH + "/LDraw/part/part_details.csv"
_PART_DETAILS_DF = pd.read_csv(_PART_DETAILS_PATH).set_index("dat_filename")

_PART_DAT_DIR = CONST.PATH + "/LDraw/part/dat/"
_PART_GEOM_DIR = CONST.PATH + "/LDraw/part/geom/"

part_by_label = {}
part_by_filename = {}
for label in _PART_LABEL_DF.index.tolist():
    dat_filename = _PART_LABEL_DF.loc[label]["dat_filename"]
    geom_filename = _PART_DETAILS_DF.loc[dat_filename]["geom_filename"]
    size_x = int(_PART_DETAILS_DF.loc[dat_filename]["size_x"])
    size_y = int(_PART_DETAILS_DF.loc[dat_filename]["size_y"])
    size_z = int(_PART_DETAILS_DF.loc[dat_filename]["size_z"])
    size = (size_x, size_y, size_z)
    ldp = LDrawPart(dat_path=_PART_DAT_DIR+dat_filename, 
                    geom_path=_PART_GEOM_DIR+geom_filename, 
                    label=label, 
                    size=size)
    part_by_label[label] = ldp
    part_by_filename[dat_filename] = ldp