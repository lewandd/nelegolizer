from nelegolizer.data import part_by_filename, ldu_to_mesh, mesh_to_ldu, part_by_id
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

    @classmethod
    def from_reference(cls, ref: LDrawReference):
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
        id = part_by_filename[ref.name].id
        return cls(id=id,
                   mesh_position=ldu_to_mesh(ref.position, id),
                   rotation=degrees,
                   color=ref.color)

    @property
    def ldu_position(self):
        return mesh_to_ldu(self.mesh_position, self.id)

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
        part_height = umesh.get_resolution(m)[1]
        height_translate = np.array([0,
                                     const.VOXEL_MESH_SHAPE[1] - part_height,
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
