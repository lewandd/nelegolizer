import numpy as np
from nelegolizer import const
from nelegolizer.utils.grid import bu_to_vu, add_padding, rotate
from typing import Tuple, List
from nelegolizer.data.voxelized_parts import stud_grid

class BrickOccupancy:
    def __init__(self, shape):
        # Brick Units occupancy
        self.shape = shape
        self.brick_grid = np.zeros(shape, dtype=bool)

        # Voxel Units occupancy (with padding)
        self.vu_shape = bu_to_vu(shape)
        self.voxel_grid = add_padding(
            np.zeros(self.vu_shape, dtype=bool), const.PADDING
        )
        
        # Stud grid
        self.stud_grid = np.zeros(shape=self.shape, dtype=bool)

        # Grid boundaries
        self.pos_min = np.zeros(3)
        self.pos_max = np.zeros(3)

    # --- PRIVATE HELPERS ---

    @staticmethod
    def _paste_subgrid(grid: np.ndarray, subgrid: np.ndarray, pos: Tuple[int, int, int]):
        """Paste subgrid to grid."""
        x, y, z = pos
        dx, dy, dz = subgrid.shape
        grid[x:x+dx, y:y+dy, z:z+dz] = subgrid

    def _grid_position(self, brick) -> np.ndarray:
        """Brick position relative to minimal position."""
        return (brick.position - self.pos_min).astype(int)
    
    def _update_studs(self, brick):
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape

        # add/remove studs from vu occupancy grid
        for x in range(pos[0], pos[0]+brick_shape[0]):
            for z in range(pos[2], pos[2]+brick_shape[2]):
                upper_position = (x, pos[1]-1, z)
                lower_position = (x, pos[1]+brick_shape[1], z)
                # usuń stud jeżeli powyżej nie ma klocka
                if not (self.in_bounds(self.brick_grid, upper_position) and self.brick_grid[upper_position]):
                    vu_pos = bu_to_vu(upper_position)+const.PADDING+np.array([0, 1, 0])
                    vu_empty = np.zeros([5,1,5], dtype=bool)
                    self._paste_subgrid(self.voxel_grid, vu_empty, vu_pos)
                # dodaj stud jeżeli poniżej jest klocek ze studem
                if self.in_bounds(self.stud_grid, lower_position) and self.stud_grid[lower_position]:
                    vu_pos = bu_to_vu(lower_position)+const.PADDING-np.array([0, 2, 0])
                    self._paste_subgrid(self.voxel_grid, stud_grid, vu_pos)

    # --- PUBLIC API ---

    def place_brick(self, brick):
        """Put a brick into brick_grid, voxel_grid and stud_grid."""
        # fill occupancy grid
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape

        # Brick Units grid
        brick_mask = np.ones(brick_shape, dtype=bool)
        self._paste_subgrid(self.brick_grid, brick_mask, pos)

        # Voxel Units grid
        stud_shape_extension = np.array([0, 1, 0])
        vu_pos = bu_to_vu(pos)+const.PADDING-stud_shape_extension
        vu_mask = rotate(brick.part.grid, brick.rotation)
        self._paste_subgrid(self.voxel_grid, vu_mask, vu_pos)

        # fill stud grid
        if brick.part.id in ["3004", "3005", "3024"]:
            self._paste_subgrid(self.stud_grid, brick_mask, pos)
    
    def get_brick_at(self, bricks: List, pos: Tuple[int, int, int]):
        """Find brick at given position."""
        for brick in bricks:
            b_pos = self._grid_position(brick)
            b_shape = brick.rotated_shape
            if all(b_pos[i] <= pos[i] < b_pos[i] + b_shape[i] for i in range(3)):
                return brick
        return None
    
    def get_brick_position(self, brick):
        return self._grid_position(brick)

    @staticmethod
    def in_bounds(grid, position):
        """Check if position is legal wtihin given grid."""
        return all(0 <= pos < dim for pos, dim in zip(position, grid.shape))

    def remove_brick(self, brick):
        """Remove a brick from brick_grid, voxel_grid and stud_grid."""
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape
        
        # Brick Units
        empty = np.zeros(brick_shape, dtype=bool)
        self._paste_subgrid(self.brick_grid, empty, pos)

        # Voxel Units
        vu_pos = bu_to_vu(pos)+const.PADDING
        vu_empty = np.zeros(bu_to_vu(brick.rotated_shape), dtype=bool)
        self._paste_subgrid(self.voxel_grid, vu_empty, vu_pos)
        self._update_studs(brick)

        # clean stud grid
        self._paste_subgrid(self.stud_grid, empty, pos)

    @classmethod
    def from_bricks(cls, bricks):
        """Create BrickOccupancy from list of bricks"""
        pos_min, pos_max = compute_bounds(bricks)
        shape = (pos_max-pos_min).astype(int)
        
        bo = cls(shape)
        bo.pos_min, bo.pos_max = pos_min, pos_max

        for brick in bricks:
            bo.place_brick(brick)
        return bo
    
def compute_bounds(lb_list):
        mins = []
        maxs = []
        
        for b in lb_list:
            b_min = np.array(b.position)
            b_max = b_min + np.array(b.rotated_shape)
            mins.append(b_min)
            maxs.append(b_max)
        
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        
        return mins, maxs