import os
import pandas as pd
from nelegolizer import path

from ._LDrawPart import LDrawPart

_PART_LABEL_DF = pd.read_csv(path.PART_LABEL_CSV).set_index("label")
_PART_DETAILS_DF = pd.read_csv(path.PART_DETAILS_CSV).set_index("dat_filename")

part_by_label = {}
part_by_filename = {}
for label in _PART_LABEL_DF.index.tolist():
    dat_filename = _PART_LABEL_DF.loc[label]["dat_filename"]
    geom_filename = _PART_DETAILS_DF.loc[dat_filename]["geom_filename"]
    size_x = int(_PART_DETAILS_DF.loc[dat_filename]["size_x"])
    size_y = int(_PART_DETAILS_DF.loc[dat_filename]["size_y"])
    size_z = int(_PART_DETAILS_DF.loc[dat_filename]["size_z"])
    size = (size_x, size_y, size_z)
    ldp = LDrawPart(dat_path=os.path.join(path.PART_DAT_DIR, dat_filename), 
                    geom_path=os.path.join(path.PART_GEOM_DIR, geom_filename), 
                    label=label, 
                    size=size)
    part_by_label[label] = ldp
    part_by_filename[dat_filename] = ldp

from ._LegoBrick import LegoBrick
from ._LDrawReference import LDrawReference
from ._LDrawModel import LDrawModel
from ._LDrawFile import LDrawFile