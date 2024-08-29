import pyvista as pv
import pandas as pd
import os
from typing import Tuple

from nelegolizer.utils import grid
from nelegolizer import const, path


class LDrawPart:
    def __init__(self, *,
                 dat_path: str,
                 geom_path: str,
                 label: int,
                 size: Tuple[int, int, int]):
        self.dat_path = dat_path
        _, self.dat_filename = os.path.split(dat_path)
        self.brick_id = self.dat_filename[:-4]
        self.geom_path = geom_path
        self.label = label
        self.size = size

        reader = pv.get_reader(geom_path)
        self.mesh = reader.read()
        self.grid = grid.from_mesh(self.mesh,
                                   voxel_mesh_shape=const.VOXEL_MESH_SHAPE)


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
