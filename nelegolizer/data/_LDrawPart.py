import pyvista as pv
import pandas as pd
import os
from typing import Tuple

from nelegolizer.utils import grid
from nelegolizer import const, path
from nelegolizer.data.voxelized_parts import part_grid, ext_part_grid
import numpy as np


class LDrawPart:
    def __init__(self, *,
                 dat_path: str,
                 geom_path: str,
                 id: str,
                 size: Tuple[int, int, int],
                 ldraw_offset: Tuple[int, int, int]):
        self.dat_path = dat_path
        _, self.dat_filename = os.path.split(dat_path)
        self.id = id
        self.geom_path = geom_path
        self.size = size
        self.ldraw_offset = ldraw_offset
        #self.ldraw_offset = np.array([0, 0, 0])
        self.ldu_offset = np.array([0, 0, 0])

        reader = pv.get_reader(geom_path)
        self.mesh = reader.read()
        self.grid = part_grid[self.id]
        self.ext_grid = ext_part_grid[self.id]


part_by_id = {}
part_by_filename = {}


def initilize_parts():
    global part_by_id
    global part_by_filename

    _PARTS2_DF = pd.read_csv(path.PARTS2_CSV).set_index("id")
    for id in _PARTS2_DF.index.tolist():
        shape = tuple(map(int, _PARTS2_DF.loc[id]["shape"].split(",")))
        ldraw_offset = tuple(map(float, _PARTS2_DF.loc[id]["ldraw_offset"].split(",")))
        dat_filename = _PARTS2_DF.loc[id]["dat_filename"]
        geom_filename = _PARTS2_DF.loc[id]["geom_filename"]
        ldu_offset = tuple(map(float, _PARTS2_DF.loc[id]["ldu_offset"].split(",")))

        ldp = LDrawPart(dat_path=os.path.join(path.PART_DAT_DIR, dat_filename),
                        geom_path=os.path.join(path.PART_GEOM_DIR, geom_filename),
                        id=str(id),
                        size=shape,
                        ldraw_offset=ldraw_offset)
        ldp.ldu_offset = ldu_offset
        part_by_id[str(id)] = ldp
        part_by_filename[str(dat_filename)] = ldp
    if (len(part_by_id) == 0) or (len(part_by_filename) == 0):
        raise Exception(f"Parts initialization failed. "
                        f"No parts found in file {path.PARTS2_CSV}.")