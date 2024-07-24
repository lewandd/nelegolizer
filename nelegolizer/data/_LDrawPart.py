import pyvista as pv
import numpy as np
from nelegolizer.utils.voxelization import voxelize_from_mesh, into_grid

class LDrawPart:
    def __init__(self, *,
                dat_path: str,
                geom_path: str,
                label: int):
        self.dat_path = dat_path
        self.geom_path = geom_path
        self.label = label
        
        reader = pv.get_reader(geom_path)
        self.__mesh = reader.read()
        voxels = voxelize_from_mesh(self.mesh, 8, 1)
        self.__grid = into_grid(voxels.cell_centers().points, 8)

    @property
    def mesh(self) -> pv.core.pointset.PolyData:
      return self.__mesh
    
    @property
    def grid(self) -> np.ndarray:
      return self.__grid