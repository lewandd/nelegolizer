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
                 id: str,
                 size: Tuple[int, int, int],
                 ldraw_offset: Tuple[int, int, int]):
        self.dat_path = dat_path
        _, self.dat_filename = os.path.split(dat_path)
        self.id = id
        self.geom_path = geom_path
        self.size = size
        self.ldraw_offset = ldraw_offset

        reader = pv.get_reader(geom_path)
        self.mesh = reader.read()
        self.grid = grid.from_mesh(self.mesh,
                                   voxel_mesh_shape=const.VOXEL_MESH_SHAPE)

_PARTS_DF = pd.read_csv(path.PARTS_CSV).set_index("id")
part_by_id = {}
part_by_filename = {}
for id in _PARTS_DF.index.tolist():
    shape = tuple(map(int, _PARTS_DF.loc[id]["shape"].split(",")))
    ldraw_offset = tuple(map(int, _PARTS_DF.loc[id]["ldraw_offset"].split(",")))
    dat_filename = _PARTS_DF.loc[id]["dat_filename"]
    geom_filename = _PARTS_DF.loc[id]["geom_filename"]

    ldp = LDrawPart(dat_path=os.path.join(path.PART_DAT_DIR, dat_filename),
                    geom_path=os.path.join(path.PART_GEOM_DIR, geom_filename),
                    id=str(id),
                    size=shape,
                    ldraw_offset=ldraw_offset)
    part_by_id[str(id)] = ldp
    part_by_filename[str(dat_filename)] = ldp