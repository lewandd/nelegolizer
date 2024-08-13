import pyvista as pv
import numpy as np
from nelegolizer.utils import voxelization
from nelegolizer.utils import grid
from nelegolizer.constants import VOXEL_UNIT_SHAPE
import os

class LDrawPart:
    def __init__(self, *,
                dat_path: str,
                geom_path: str,
                label: int,
                size: tuple[int, int, int]):
        self.dat_path = dat_path
        _, self.dat_filename = os.path.split(dat_path)
        self.geom_path = geom_path
        self.label = label
        self.size = size
        
        reader = pv.get_reader(geom_path)
        self.mesh: pv.PolyData = reader.read()
        self.grid: np.ndarray = grid.from_mesh(self.mesh, unit_shape=VOXEL_UNIT_SHAPE)