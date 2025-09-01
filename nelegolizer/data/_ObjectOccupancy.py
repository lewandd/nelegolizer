import numpy as np
from nelegolizer import const
from nelegolizer.utils.grid import vu_to_bu, bu_to_vu
import nelegolizer.utils.grid as grid

class ObjectOccupancy:
    def __init__(self, voxel_grid):
        # Voxel Units occupancy (with padding)
        self.vu_shape = voxel_grid.shape
        self.voxel_grid = voxel_grid
        
        # Initialize Brick Unit brick_grid
        self.shape = vu_to_bu(self.vu_shape-2*const.PADDING)
        self.brick_grid = np.zeros(self.shape, dtype=bool)
        for pos, _ in np.ndenumerate(self.brick_grid):
            x, y, z = pos
            self.brick_grid[x, y, z] = grid.get_fill(
                # TODO zmienione
                #grid.get_subgrid(voxel_grid, bu_to_vu(pos)+const.PADDING+np.array([0, 1, 0]), const.BRICK_UNIT_RESOLUTION)) > 0
                grid.get_subgrid(voxel_grid, bu_to_vu(pos)+const.PADDING, const.BRICK_UNIT_RESOLUTION)) > 0