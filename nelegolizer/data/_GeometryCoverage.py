import numpy as np
from nelegolizer import const
from nelegolizer.utils.grid import vu_to_bu, bu_to_vu
import nelegolizer.utils.grid as grid
from typing import Tuple, Union

EXTENSION = np.array([1, 1, 1])

def ext_bu_to_vu(bu: Union[np.ndarray, Tuple[int, int, int]]) -> np.ndarray:
    arr = np.array(bu, dtype=int)

    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape} from {bu!r}")

    result = arr * (const.BRICK_UNIT_RESOLUTION + EXTENSION)

    if isinstance(bu, tuple):
        return tuple(result.tolist())
    return result

class GeometryCoverage:
    def __init__(self, voxel_grid):
        # Voxel Units occupancy (with padding)
        self.vu_shape = voxel_grid.shape
        self.voxel_grid = voxel_grid
        print("original voxel_shape", voxel_grid.shape)

        # Initialize Brick Unit brick_grid
        self.shape = vu_to_bu(voxel_grid.shape-2*const.PADDING)
        print("brick shape", self.shape)
        self.brick_grid = np.zeros(self.shape, dtype=bool)
        for pos, _ in np.ndenumerate(self.brick_grid):
            x, y, z = pos
            vu_pos = bu_to_vu(pos)+const.PADDING
            self.brick_grid[x, y, z] = grid.get_fill(
                grid.get_subgrid(voxel_grid, vu_pos, const.BRICK_UNIT_RESOLUTION)) > 0
            

        # extended Voxel Units
        self.ext_vu_shape = ext_bu_to_vu(self.shape)+2*const.PADDING
        self.ext_voxel_grid = np.zeros(self.ext_vu_shape, dtype=bool)

        # wypełnianie wokselami wnętrza komórek
        for pos, _ in np.ndenumerate(self.brick_grid):
            x, y, z = pos
            ext_vu_pos = ext_bu_to_vu(pos)+const.PADDING+np.array([0, 1, 0])
            vu_pos = bu_to_vu(pos)+const.PADDING
            vu_mask = grid.get_subgrid(voxel_grid, vu_pos, const.BRICK_UNIT_RESOLUTION)
            self._paste_subgrid(self.ext_voxel_grid, vu_mask, ext_vu_pos)
            

        # wypełnianie wokselami przestrzeniami między komórkami
        for _ in range(3):
            for pos, _ in np.ndenumerate(self.brick_grid):
                x, y, z = pos
                if self.brick_grid[x, y, z]:

                    ext_vu_pos = ext_bu_to_vu(pos)+const.PADDING
                    vu_shape = const.BRICK_UNIT_RESOLUTION

                    for i in range(vu_shape[0]+1):
                        for k in range(vu_shape[2]+1):
                            check_pos = (ext_vu_pos[0]+i, ext_vu_pos[1], ext_vu_pos[2]+k)
                            upper_pos = (check_pos[0], check_pos[1]-1, check_pos[2])
                            lower_pos = (check_pos[0], check_pos[1]+1, check_pos[2])
                            if self.ext_voxel_grid[lower_pos] and self.ext_voxel_grid[upper_pos]:
                                self.ext_voxel_grid[check_pos] = True      

                    for i in range(vu_shape[0]+1):
                        for j in range(vu_shape[1]+1):
                            check_pos = (ext_vu_pos[0]+i, ext_vu_pos[1]+j, ext_vu_pos[2]+5)
                            side1z_pos = (check_pos[0], check_pos[1], check_pos[2]+1)
                            side2z_pos = (check_pos[0], check_pos[1], check_pos[2]-1)
                            if self.ext_voxel_grid[side1z_pos] and self.ext_voxel_grid[side2z_pos]:
                                self.ext_voxel_grid[check_pos] = True  
                    
                    for j in range(vu_shape[1]+1):
                        for k in range(vu_shape[2]+1):
                            check_pos = (ext_vu_pos[0]+5, ext_vu_pos[1]+j, ext_vu_pos[2]+k)
                            side1x_pos = (check_pos[0]-1, check_pos[1], check_pos[2])
                            side2x_pos = (check_pos[0]+1, check_pos[1], check_pos[2])
                            if self.ext_voxel_grid[side1x_pos] and self.ext_voxel_grid[side2x_pos]:
                                self.ext_voxel_grid[check_pos] = True  


    @staticmethod
    def _paste_subgrid(grid: np.ndarray, subgrid: np.ndarray, pos: Tuple[int, int, int]):
        """Paste subgrid to grid."""
        x, y, z = pos
        dx, dy, dz = subgrid.shape
        grid[x:x+dx, y:y+dy, z:z+dz] = subgrid