import pyvista as pv
import numpy as np
from nelegolizer.utils import voxelization
from nelegolizer.utils import grid

class LDrawPart:
    def __init__(self, *,
                dat_path: str,
                geom_path: str,
                label: int,
                size: tuple[int, int, int]):
        self.dat_path = dat_path
        self.geom_path = geom_path
        self.label = label
        self.size = size
        
        reader = pv.get_reader(geom_path)
        self.__mesh = reader.read()
        pv_voxels = voxelization.from_mesh(self.mesh, 8, 1)
        self.__grid = grid.from_pv_voxels(pv_voxels, 8)

    @property
    def mesh(self) -> pv.core.pointset.PolyData:
      return self.__mesh
    
    @property
    def grid(self) -> np.ndarray:
      return self.__grid