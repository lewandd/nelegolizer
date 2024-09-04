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


_PART_DATA_DF = pd.read_csv(path.PART_DATA_CSV).set_index("dat_filename")

part_by_size_label = {}
part_by_filename = {}
for dat_filename in _PART_DATA_DF.index.tolist():
    label = _PART_DATA_DF.loc[dat_filename]["label"]
    geom_filename = _PART_DATA_DF.loc[dat_filename]["geom_filename"]
    size_x = int(_PART_DATA_DF.loc[dat_filename]["size_x"])
    size_y = int(_PART_DATA_DF.loc[dat_filename]["size_y"])
    size_z = int(_PART_DATA_DF.loc[dat_filename]["size_z"])
    size = (size_x, size_y, size_z)
    ldp = LDrawPart(dat_path=os.path.join(path.PART_DAT_DIR, dat_filename),
                    geom_path=os.path.join(path.PART_GEOM_DIR, geom_filename),
                    label=label,
                    size=size)
    if not str(size) in part_by_size_label:
        part_by_size_label[str(size)] = {}
    part_by_size_label[str(size)][label] = ldp

    part_by_filename[dat_filename] = ldp
