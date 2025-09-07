import numpy as np
from nelegolizer import const
from nelegolizer.utils.conversion import *
from nelegolizer.utils.grid import get_subgrid, get_fill
from typing import Tuple

class GeometryCoverage:
    def __init__(self, voxel_grid, bottom_extension=0, top_extension=0, side_extension=0):
        self.BOT_EXT = bottom_extension
        self.TOP_EXT = top_extension
        self.VERT_EXT = self.BOT_EXT + self.TOP_EXT
        self.SIDE_EXT = side_extension

        # Voxel Units occupancy (with padding)
        
        self.voxel_grid = np.pad(voxel_grid,
                                 pad_width=((side_extension*5, side_extension*5),
                                            (top_extension*2, bottom_extension*2),
                                            (side_extension*5, side_extension*5)),
                                 mode="constant",
                                 constant_values=0)
        self.vu_shape = self.voxel_grid.shape
        print(f"GeometryCoverage: voxel_grid.shape = {self.voxel_grid.shape}")

        # Initialize Brick Unit brick_grid
        self.interior_shape = vu_to_bu(voxel_grid.shape)
        self.shape = self.interior_shape + np.array([self.SIDE_EXT*2, self.VERT_EXT, self.SIDE_EXT*2])
        self.brick_grid = np.zeros(self.shape, dtype=bool)
        print(f"GeometryCoverage: brick_grid.shape = {self.brick_grid.shape}")
        for pos, _ in np.ndenumerate(self.brick_grid):
        #for pos, _ in np.ndenumerate(self.brick_grid[self.SIDE_EXT:-self.SIDE_EXT,self.TOP_EXT:-self.BOT_EXT,self.SIDE_EXT:-self.SIDE_EXT]):
            x, y, z = pos
            ext_pos = pos# + np.array([self.SIDE_EXT, self.TOP_EXT, self.SIDE_EXT])
            vu_pos = bu_to_vu(pos)+const.PADDING
            
            self.brick_grid[ext_pos[0], ext_pos[1], ext_pos[2]] = get_fill(
                get_subgrid(self.voxel_grid, vu_pos, const.BRICK_UNIT_RESOLUTION)) > 0
            

        # extended Voxel Units
        self.ext_vu_shape = ext_bu_to_vu(np.array(self.shape))# + np.array([self.SIDE_EXT*2, self.VERT_EXT, self.SIDE_EXT*2])
        #self.ext_vu_shape = ext_bu_to_vu(self.shape)+2*const.PADDING
        self.ext_voxel_grid = np.zeros(self.ext_vu_shape, dtype=bool)
        print(f"GeometryCoverage: ext_voxel_grid.shape = {self.ext_voxel_grid.shape}")

        ext_ones = np.ones(shape=(self.ext_vu_shape[0], const.PADDING[1], self.ext_vu_shape[2]))
        ext_ones_pos = (0, self.ext_vu_shape[1]-const.PADDING[1], 0)
        self._paste_subgrid(self.ext_voxel_grid, ext_ones, ext_ones_pos)

        if self.BOT_EXT:
            # bottom
            ext_ones = np.ones(shape=(self.ext_vu_shape[0], self.BOT_EXT*EXTBU[1]+const.PADDING[1], self.ext_vu_shape[2]))
            ext_ones_pos = (0, self.ext_vu_shape[1]-const.PADDING[1]-self.BOT_EXT*EXTBU[1], 0)
            self._paste_subgrid(self.ext_voxel_grid, ext_ones, ext_ones_pos)
            #for x in range(self.shape[0]):
            #    for z in range(self.shape[2]):
            #        y = self.shape[1] - self.BOT_EXT
            #        ext_vu_pos = ext_bu_to_vu(np.array([x, y, z]))+const.PADDING-np.array([0, 2, 0])
            #        self._paste_subgrid(self.ext_voxel_grid, ext_stud_grid, ext_vu_pos)

        # wypełnianie wokselami wnętrza komórek
        #for pos, _ in np.ndenumerate(self.brick_grid[self.SIDE_EXT:-self.SIDE_EXT,self.TOP_EXT:-self.BOT_EXT,self.SIDE_EXT:-self.SIDE_EXT]):
        #    x, y, z = pos
        #    ext_pos = pos + np.array([self.SIDE_EXT, self.TOP_EXT, self.SIDE_EXT])
        #    if self.brick_grid[ext_pos[0], ext_pos[1], ext_pos[2]]:
        #        
        #        #ext_vu_pos = ext_bu_to_vu(pos)+const.PADDING+np.array([0, 1, 0])
        #        ext_vu_pos = ext_bu_to_vu(ext_pos)+const.PADDING+np.array([0, 1, 0])
        #        # dodałem ext_pos zamiast pos
        #        vu_pos = bu_to_vu(ext_pos)+const.PADDING
        #        vu_mask = grid.get_subgrid(voxel_grid, vu_pos, const.BRICK_UNIT_RESOLUTION)
        #        self._paste_subgrid(self.ext_voxel_grid, vu_mask, ext_vu_pos)

        # wypełnianie wokselami wnętrza komórek
        for pos, _ in np.ndenumerate(self.brick_grid):
            x, y, z = pos
            ext_pos = pos# + np.array([self.SIDE_EXT, self.TOP_EXT, self.SIDE_EXT])
            if self.brick_grid[ext_pos[0], ext_pos[1], ext_pos[2]]:
                
                #ext_vu_pos = ext_bu_to_vu(pos)+const.PADDING+np.array([0, 1, 0])
                ext_vu_pos = ext_bu_to_vu(ext_pos)+const.PADDING+np.array([0, 1, 0])
                # dodałem ext_pos zamiast pos
                vu_pos = bu_to_vu(ext_pos)+const.PADDING
                vu_mask = get_subgrid(self.voxel_grid, vu_pos, const.BRICK_UNIT_RESOLUTION)
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
                            if True:#in_bounds(self.ext_voxel_grid, lower_pos) and in_bounds(self.ext_voxel_grid, upper_pos):
                                if self.ext_voxel_grid[lower_pos] and self.ext_voxel_grid[upper_pos]:
                                    self.ext_voxel_grid[check_pos] = True      

                    for i in range(vu_shape[0]+1):
                        for j in range(vu_shape[1]+1):
                            check_pos = (ext_vu_pos[0]+i, ext_vu_pos[1]+j, ext_vu_pos[2]+5)
                            side1z_pos = (check_pos[0], check_pos[1], check_pos[2]+1)
                            side2z_pos = (check_pos[0], check_pos[1], check_pos[2]-1)
                            if  True:#in_bounds(self.ext_voxel_grid, side1z_pos) and in_bounds(self.ext_voxel_grid, side2z_pos):
                                if self.ext_voxel_grid[side1z_pos] and self.ext_voxel_grid[side2z_pos]:
                                    self.ext_voxel_grid[check_pos] = True  
                    
                    for j in range(vu_shape[1]+1):
                        for k in range(vu_shape[2]+1):
                            check_pos = (ext_vu_pos[0]+5, ext_vu_pos[1]+j, ext_vu_pos[2]+k)
                            side1x_pos = (check_pos[0]-1, check_pos[1], check_pos[2])
                            side2x_pos = (check_pos[0]+1, check_pos[1], check_pos[2])
                            if  True:#in_bounds(self.ext_voxel_grid, side1x_pos) and in_bounds(self.ext_voxel_grid, side2x_pos):
                                if self.ext_voxel_grid[side1x_pos] and self.ext_voxel_grid[side2x_pos]:
                                    self.ext_voxel_grid[check_pos] = True  


    @staticmethod
    def _paste_subgrid(grid: np.ndarray, subgrid: np.ndarray, pos: Tuple[int, int, int]):
        """Paste subgrid to grid."""
        x, y, z = pos
        dx, dy, dz = subgrid.shape
        grid[x:x+dx, y:y+dy, z:z+dz] = subgrid