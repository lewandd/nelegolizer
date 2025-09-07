from nelegolizer.data import part_by_filename, part_by_id
from nelegolizer.utils.conversion import *
from nelegolizer.utils import mesh as umesh
from nelegolizer.utils import grid
from nelegolizer.data import LDrawReference
from nelegolizer import const
from typing import Tuple
import constants as const

import numpy as np

ROT_MATRIX_0 = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
ROT_MATRIX_90 = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
ROT_MATRIX_180 = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
ROT_MATRIX_270 = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 0]])


class LegoBrick:
    def __init__(self, *,
                 id: str,
                 mesh_position: Tuple[int, int, int],
                 rotation: int,
                 color: int = 16):
        # part
        try:
            self.part = part_by_id[id]
        except KeyError:
            available_keys = list(part_by_id.keys())
            raise KeyError(f"LegoBrick id can be: {available_keys}. "
                           f"Brick id {id} is invalid.")

        # position
        self.mesh_position = mesh_position

        self.real_position = None

        #rotation
        valid_rotations = (0, 90, 180, 270)
        if rotation not in valid_rotations:
            raise Exception(
                f"LegoBrick rotation can be either {valid_rotations}." 
                f"Rotation {rotation} is invalid.")
        self.rotation = rotation

        # color
        self.color = color

    @property
    def id(self) -> str:
        return self.part.id
    
    @property
    def position(self):
        pos = mesh_to_bu(self.mesh_position)
        #if self.id == "54200":
        #    return pos - np.array([0, 2, 0])
        return pos

    @property
    def rotated_shape(self) -> str:
        if self.rotation == 0 or self.rotation == 180:
            return self.part.size
        elif self.rotation == 90 or self.rotation == 270:
            return (self.part.size[2], self.part.size[1], self.part.size[0])

    @classmethod
    def from_reference(cls, ref: LDrawReference):
        # find rotation matrix
        if np.allclose(ref.rotation, ROT_MATRIX_0):
            degrees = 0
        elif np.allclose(ref.rotation, ROT_MATRIX_90):
            degrees = 90
        elif np.allclose(ref.rotation, ROT_MATRIX_180):
            degrees = 180
        elif np.allclose(ref.rotation, ROT_MATRIX_270):
            degrees = 270
        else:
            raise Exception(f"Cannot convert model references to bricks."
                            f" Rotation should be either: \n>{ROT_MATRIX_0}\n"
                            f">{ROT_MATRIX_90}\n>{ROT_MATRIX_180}\n"
                            f">{ROT_MATRIX_270}\nGot: \n{ref.rotation}.")
        
        # get a part
        try:
            part = part_by_filename[ref.name]
        except KeyError:
            if len(part_by_filename) == 0:
                raise KeyError(f"No filenames in part_by_filename. "
                           f"Initizalize parts with function initilize_parts() "
                           f"from nelegolizer.data.")    
            else:
                raise KeyError(f"No {ref.name} filename in part_by_filename. "
                           f"Available filenames: {part_by_filename.keys()}")

        # rotate offset
        #if degrees in [0, 180]:
        #    ldraw_offset = part.ldraw_offset
        #elif degrees in [90, 270]:
        #    ldraw_offset = (part.ldraw_offset[2], part.ldraw_offset[1], part.ldraw_offset[0])
        
        if degrees in [0, 180]:
            ldu_offset = part.ldu_offset
        elif degrees in [90, 270]:
            ldu_offset = (part.ldu_offset[2], part.ldu_offset[1], part.ldu_offset[0])

        # TUTAJ ZMIENIA SIĘ POZYCJA
        #print(ref.position, "and", ldu_offset)
        return cls(id=part.id,
                   #mesh_position=ldu_to_mesh(ref.position)-ldraw_offset,
                   mesh_position=ldu_to_mesh(ref.position-ldu_offset),
                   rotation=degrees,
                   color=ref.color)

    # TODO: kiedy jest konwertowane LegoBrick do LDraw to powinno się przywrócić
    #       ten offset, który teraz został na stałe odjęty

    @property
    def ldu_position(self):
        return mesh_to_ldu(self.mesh_position, self.id, self.rotation)

    @property
    def matrix(self):
        rotation = np.zeros([4, 4])
        rotation[-1, :3] = self.ldu_position
        rotation[-1, -1] = 1
        if self.rotation == 0:
            rotation[:3, :3] = ROT_MATRIX_0
        elif self.rotation == 90:
            rotation[:3, :3] = ROT_MATRIX_90
        elif self.rotation == 180:
            rotation[:3, :3] = ROT_MATRIX_180
        elif self.rotation == 270:
            rotation[:3, :3] = ROT_MATRIX_270
        return rotation

    @property
    def mesh(self):
        m = self.part.mesh
        m = m.rotate_y(angle=self.rotation, inplace=False)
        m = umesh.translate_to_zero(m)
        
        # translate to make mesh (0,0,0) pos to upper BU bounding box corner
        part_height = umesh.get_resolution(m)[1]
        height_translate = np.array([0,
                                     self.rotated_shape[1]*2*const.VOXEL_MESH_SHAPE[1] - part_height,
                                     0])
        m = m.translate(height_translate, inplace=False)
        
        m = m.translate(self.mesh_position, inplace=False)
        return m

    @property
    def grid(self):
        g = self.part.grid
        g = grid.rotate(g, self.rotation)
        return g

    def __str__(self):
        string = "LegoBrick: "
        string += "dat_filename=" + str(self.part.dat_filename) + ", "
        string += "position=" + str(self.mesh_position) + ", "
        string += "rotation=" + str(self.rotation)
        string += "color=" + str(self.color)
        return string
