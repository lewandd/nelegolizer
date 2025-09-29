from ..utils import grid as utils_grid
from ..utils import brick as utils_brick
from ..utils.conversion import ext_bu_to_vu, bu_to_vu
from ..constants import EXT_BU_RES
from .voxelized_parts import stud_grid, ext_part_grid2, ext_stud_grid
from typing import Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BrickCoverage:
    def __init__(self, shape, bottom_extension=0, top_extension=0, side_extension=0):
        self.BOT_EXT = bottom_extension
        self.TOP_EXT = top_extension
        self.VERT_EXT = self.BOT_EXT + self.TOP_EXT
        self.SIDE_EXT = side_extension

        # Brick Units
        self.shape = np.array(shape) + np.array([self.SIDE_EXT*2, self.VERT_EXT, self.SIDE_EXT*2])
        self.brick_grid = np.zeros(self.shape, dtype=bool)
        logger.debug(f"BrickCoverage: brick_grid.shape = {self.brick_grid.shape}")

        # Voxel Units
        self.vu_shape = bu_to_vu(self.shape)
        self.voxel_grid = np.zeros(self.vu_shape, dtype=bool)

        # ???
        #ones = np.ones(shape=(self.vu_shape[0], const.PADDING[1], self.vu_shape[2]))
        #ones_pos = (0, self.vu_shape[1], 0)
        #self._paste_subgrid(self.voxel_grid, ones, ones_pos)

        # extended Voxel Units
        self.ext_vu_shape = ext_bu_to_vu(self.shape)
        #print(self.ext_vu_shape)
        self.ext_voxel_grid = np.zeros(self.ext_vu_shape, dtype=bool)
        logger.debug(f"BrickCoverage: ext_voxel_grid.shape = {self.ext_voxel_grid.shape}")
        if self.BOT_EXT:
            # bottom
            ext_ones = np.ones(shape=(self.ext_vu_shape[0], self.BOT_EXT*EXT_BU_RES[1]-1, self.ext_vu_shape[2]))
            ext_ones_pos = (0, self.ext_vu_shape[1]-self.BOT_EXT*EXT_BU_RES[1]+1, 0)
            self._paste_subgrid(self.ext_voxel_grid, ext_ones, ext_ones_pos)
            for x in range(self.shape[0]):
                for z in range(self.shape[2]):
                    y = self.shape[1] - self.BOT_EXT
                    ext_vu_pos = ext_bu_to_vu(np.array([x, y, z]))-np.array([0, 2, 0])
                    self._paste_subgrid(self.ext_voxel_grid, ext_stud_grid, ext_vu_pos)

        # Stud grid
        self.stud_grid = np.zeros(shape=self.shape, dtype=bool)

        # top connection availibility
        self.top_available_grid = np.zeros(shape=self.shape, dtype=bool)
        floor_availible = np.ones((self.shape[0], 1, self.shape[2]), dtype=bool)
        floor_available_pos = (0, self.shape[1]-self.BOT_EXT-1, 0)
        self._paste_subgrid(self.top_available_grid, floor_availible, floor_available_pos)

        # bottom connection availibility
        self.bottom_available_grid = np.zeros(shape=self.shape, dtype=bool)

        # bottom connection availibility
        self.bottom3_available_grid = np.zeros(shape=self.shape, dtype=bool)

        # Grid boundaries
        self.pos_min = np.zeros(3)
        self.pos_max = np.zeros(3)

    # --- PRIVATE HELPERS ---

    @staticmethod
    def _paste_subgrid(grid: np.ndarray, subgrid: np.ndarray, pos: Tuple[int, int, int]):
        """Paste subgrid to grid."""
        x, y, z = pos
        dx, dy, dz = subgrid.shape
        try:
            grid[x:x+dx, y:y+dy, z:z+dz] = subgrid
        except ValueError:
            raise ValueError(f"tried paste {subgrid.shape} subgrid into position {pos} of {grid.shape} grid.")

    def _grid_position(self, brick) -> np.ndarray:
        """Brick position relative to minimal position."""
        return np.round(brick.position - self.pos_min).astype(int)
    
    def _update_studs(self, brick):
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape

        # add/remove studs from vu occupancy grid
        for x in range(pos[0], pos[0]+brick_shape[0]):
            for z in range(pos[2], pos[2]+brick_shape[2]):
                upper_position = (x, pos[1]-1, z)
                lower_position = (x, pos[1]+brick_shape[1]+1, z)
                # usuń stud jeżeli powyżej nie ma klocka
                if not (self.in_bounds(self.brick_grid, upper_position) and self.brick_grid[upper_position]):
                    vu_pos = bu_to_vu(upper_position)+np.array([0, 3, 0])
                    vu_empty = np.zeros([5,1,5], dtype=bool)
                    self._paste_subgrid(self.voxel_grid, vu_empty, vu_pos)
                # dodaj stud jeżeli poniżej jest klocek ze studem
                if self.in_bounds(self.stud_grid, lower_position) and self.stud_grid[lower_position]:
                    vu_pos = bu_to_vu(lower_position)-np.array([0, 2, 0])
                    self._paste_subgrid(self.voxel_grid, stud_grid, vu_pos)

    def _ext_upper_pos(self, brick):
        brick_shape = brick.rotated_shape
        pos = self._grid_position(brick)
        upper_positions = []

        for x in range(pos[0], pos[0]+brick_shape[0]):
            for z in range(pos[2], pos[2]+brick_shape[2]):
                upper_position = (x, pos[1]-1, z)
                if self.in_bounds(self.brick_grid, upper_position):
                    ext_vu_pos = ext_bu_to_vu(upper_position)+np.array([0, 5, 0])
                    upper_positions.append(ext_vu_pos)
        return upper_positions

    def _ext_lower_pos(self, brick):
        brick_shape = brick.rotated_shape
        pos = self._grid_position(brick)
        lower_positions = []

        for x in range(pos[0], pos[0]+brick_shape[0]):
            for z in range(pos[2], pos[2]+brick_shape[2]):
                lower_position = (x, pos[1]+brick_shape[1]+1, z)
                if self.in_bounds(self.brick_grid, lower_position):
                    ext_vu_pos = ext_bu_to_vu(lower_position)-np.array([0, 2, 0])
                    lower_positions.append(ext_vu_pos)
        return lower_positions

    def _ext_update_studs(self, brick):
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape

        # add/remove studs from vu occupancy grid
        for x in range(pos[0], pos[0]+brick_shape[0]):
            for z in range(pos[2], pos[2]+brick_shape[2]):
                upper_position = (x, pos[1]-1, z)
                lower_position = (x, pos[1]+brick_shape[1]+1, z) # lower position powinno być dużo niżej
                # usuń stud jeżeli powyżej nie ma klocka
                if not (self.in_bounds(self.brick_grid, upper_position) and self.brick_grid[upper_position]):
                    vu_pos = ext_bu_to_vu(upper_position)+np.array([0, 5, 0])
                    vu_empty = np.zeros([5,2,5], dtype=bool)
                    self._paste_subgrid(self.ext_voxel_grid, vu_empty, vu_pos)
                # dodaj stud jeżeli poniżej jest klocek ze studem
                if self.in_bounds(self.stud_grid, lower_position) and self.stud_grid[lower_position]:
                    vu_pos = ext_bu_to_vu(lower_position)-np.array([0, 2, 0])
                    self._paste_subgrid(self.ext_voxel_grid, ext_stud_grid, vu_pos)

    # --- PUBLIC API ---

    def is_placement_available(self, brick):
        ext_pos = self._grid_position(brick)+ np.array([0, 1, 0])# + np.array([self.SIDE_EXT, self.TOP_EXT, self.SIDE_EXT]) 
        brick_shape = brick.rotated_shape
        placement = utils_grid.get_subgrid(self.brick_grid, ext_pos, brick_shape)
        return utils_grid.is_empty(placement)
        
    def place_brick(self, brick, debug=False):
        """Put a brick into brick_grid, voxel_grid and stud_grid."""
        # fill occupancy grid
        ext_pos = self._grid_position(brick) + np.array([0, 1, 0]) #+ np.array([self.SIDE_EXT, self.TOP_EXT, self.SIDE_EXT])
        pos = self._grid_position(brick)
        brick_shape = brick.rotated_shape
        if debug:
            print(f"self.brick_grid filled with brick at pos: {ext_pos} with shape: {brick_shape}")

        # Brick Units grid
        brick_mask = np.ones(brick_shape, dtype=bool)
        self._paste_subgrid(self.brick_grid, brick_mask, ext_pos)

        # Voxel Units grid
        #stud_shape_extension = np.array([0, 1, 0])
        vu_pos = bu_to_vu(pos)+np.array([0, 1, 0])
        vu_mask = utils_grid.rotate(brick.part.grid, brick.rotation)
        self._paste_subgrid(self.voxel_grid, vu_mask, vu_pos)

        # update above Voxel Unit grid
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                used_available_pos = (ext_pos[0]+x, ext_pos[1], ext_pos[2]+z)
                if self.bottom_available_grid[used_available_pos]:
                    vu_pos = bu_to_vu(pos)+np.array([0, 1, 0])
                    w, _, d = ext_bu_to_vu((1, 1, 1))
                    vu_ones = np.ones((w-1, 1, d-1), dtype=bool)
                    self._paste_subgrid(self.voxel_grid, vu_ones, vu_pos)

        # extended Voxel Units griD
        ext_vu_pos = ext_bu_to_vu(ext_pos)-np.array([0, 1, 0])
        ext_vu_mask = ext_part_grid2[brick.part.id][brick.rotation]
        self._paste_subgrid(self.ext_voxel_grid, ext_vu_mask, ext_vu_pos)

        # update above extended Voxel Unit grid
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                used_available_pos = (ext_pos[0]+x, ext_pos[1], ext_pos[2]+z)
                if self.bottom_available_grid[used_available_pos]:
                    ext_vu_pos = ext_bu_to_vu(ext_pos)-np.array([0, 1, 0])
                    w, _, d = ext_bu_to_vu((1, 1, 1))
                    ext_vu_ones = np.ones((w-1, 1, d-1), dtype=bool)
                    self._paste_subgrid(self.ext_voxel_grid, ext_vu_ones, ext_vu_pos)

        # update connections
        available_mask = np.ones((1, 1, 1), dtype=bool)
        unavailable_mask = ~available_mask

        # add top connection availibility
        if brick.part.id in ["3004", "3005", "3024", "3023"]:
            for x in range(brick_shape[0]):
                for z in range(brick_shape[2]):
                    above_available_pos = (ext_pos[0]+x, ext_pos[1]-1, ext_pos[2]+z)
                    if not self.brick_grid[above_available_pos]: # if above position is free
                        self._paste_subgrid(self.top_available_grid, available_mask, above_available_pos)

        # remove used connection availibility from below
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                used_available_pos = (ext_pos[0]+x, ext_pos[1]+brick_shape[1]-1, ext_pos[2]+z)
                self._paste_subgrid(self.top_available_grid, unavailable_mask, used_available_pos)

        # add bottom connection availibility
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                below_available_pos = (ext_pos[0]+x, ext_pos[1]+brick_shape[1], ext_pos[2]+z)
                if not self.brick_grid[below_available_pos] and below_available_pos[1] < self.shape[1]-self.BOT_EXT: # if above position is free
                    self._paste_subgrid(self.bottom_available_grid, available_mask, below_available_pos)


        # --------------------------------------- IN WORK
        # add bottom3 connection availibility
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                below_available_pos = (ext_pos[0]+x, ext_pos[1]+brick_shape[1], ext_pos[2]+z)
                below2_available_pos = (ext_pos[0]+x, ext_pos[1]+brick_shape[1]+1, ext_pos[2]+z)
                below3_available_pos = (ext_pos[0]+x, ext_pos[1]+brick_shape[1]+2, ext_pos[2]+z)
                below_occupied = (self.brick_grid[below_available_pos] or 
                                  self.brick_grid[below2_available_pos] or
                                  self.brick_grid[below3_available_pos])
                if not below_occupied and below3_available_pos[1] < self.shape[1]-self.BOT_EXT: # if above position is free
                    self._paste_subgrid(self.bottom3_available_grid, available_mask, below3_available_pos)
        # ------------------------------------------------

        # remove used connection availibility from above
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                used_available_pos = (ext_pos[0]+x, ext_pos[1], ext_pos[2]+z)
                self._paste_subgrid(self.bottom_available_grid, unavailable_mask, used_available_pos)

        # ------------------------------------------- IN WORK
        # remove used 3-connection availibility from above
        #if brick_shape[1] >= 3:
        for x in range(brick_shape[0]):
            for z in range(brick_shape[2]):
                used_available_pos = (ext_pos[0]+x, ext_pos[1], ext_pos[2]+z)
                used_available_pos3 = (ext_pos[0]+x, ext_pos[1]+2, ext_pos[2]+z)
                self._paste_subgrid(self.bottom3_available_grid, unavailable_mask, used_available_pos3)
        # ----------------------------------------------------

        # fill stud grid
        if brick.part.id in ["3004", "3005", "3024", "3023"]:
            self._paste_subgrid(self.stud_grid, brick_mask, pos)

    def get_brick_at(self, bricks: List, pos: Tuple[int, int, int]):
        """Find brick at given position."""
        for brick in bricks:
            b_pos = self._grid_position(brick)
            # TUTAJ?
            #b_pos = np.array([b_pos[0] + self.SIDE_EXT, b_pos[1] + self.TOP_EXT, b_pos[2] + self.SIDE_EXT,])
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

        # extended Voxel Units
        interior_pos = ext_bu_to_vu(pos+np.array([0, 1, 0])) + np.array([0, 1, 0])
        interior_empty_shape = ext_part_grid2[brick.part.id][brick.rotation].shape - np.array([0, 4, 0])
        interior_empty = np.zeros(shape=interior_empty_shape, dtype=bool)
        
        upper_connection_pos = interior_pos - np.array([0, 1, 0])
        lower_connection_pos = ext_bu_to_vu(pos+np.array([0, 4, 0]))
        connection_empty_shape = (interior_empty_shape[0], 1, interior_empty_shape[2])
        connection_empty = np.zeros(shape=connection_empty_shape, dtype=bool)

        
        self._paste_subgrid(self.ext_voxel_grid, interior_empty, interior_pos)
        self._paste_subgrid(self.ext_voxel_grid, connection_empty, upper_connection_pos)
        if len(self._ext_lower_pos(brick)):
            #print("----- REMOVED LOWER CONNECTION -----")
            #print(self._ext_lower_pos(brick))
            #print("------------------------------------")
            self._paste_subgrid(self.ext_voxel_grid, connection_empty, lower_connection_pos)
        self._ext_update_studs(brick)


        # Voxel Units
        vu_pos = bu_to_vu(pos)+np.array([0, 2, 0])
        vu_empty = np.zeros(bu_to_vu(brick.rotated_shape), dtype=bool)
        self._paste_subgrid(self.voxel_grid, vu_empty, vu_pos)
        self._update_studs(brick)

        # Voxel Units grid
        #stud_shape_extension = np.array([0, 1, 0])
        #vu_pos = mod_bu_to_vu(pos)+const.PADDING-stud_shape_extension
        #vu_mask = ext_part_grid2[brick.part.id][brick.rotation]
        #self._paste_subgrid(self.voxel_grid, vu_mask, vu_pos)

        # clean stud grid
        self._paste_subgrid(self.stud_grid, empty, pos)

    @classmethod
    def from_bricks(cls, bricks, bottom_extension=0, top_extension=0, side_extension=0, shape=None):
        """Create BrickOccupancy from list of bricks"""
        pos_min, pos_max = utils_brick.compute_bounds(bricks)
        
        if shape is None:
            shape = (pos_max-pos_min).astype(int) + np.array([0, 1, 0])
        else:
            shape = shape+np.array([0, 1, 0])

        bo = cls(shape, bottom_extension, top_extension, side_extension) 
        #bo.pos_min, bo.pos_max = pos_min, pos_max
        bo.pos_min, bo.pos_max = 0, 0

        for brick in bricks:
            bo.place_brick(brick)
        return bo
    