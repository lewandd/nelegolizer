import pyvista as pv
import numpy as np
from nelegolizer.utils import voxelization
from nelegolizer.utils import grid
from nelegolizer import const
from typing import Tuple
import os

class LDrawPart:
    def __init__(self, *,
                dat_path: str,
                geom_path: str,
                label: int,
                size: Tuple[int, int, int]):
        self.dat_path = dat_path
        _, self.dat_filename = os.path.split(dat_path)
        self.geom_path = geom_path
        self.label = label
        self.size = size
        
        reader = pv.get_reader(geom_path)
        self.mesh: pv.PolyData = reader.read()
        self.grid: np.ndarray = grid.from_mesh(self.mesh, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)