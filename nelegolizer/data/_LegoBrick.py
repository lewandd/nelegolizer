from nelegolizer.data import part_by_filename
from nelegolizer.utils import mesh as umesh
from nelegolizer.utils import grid
from nelegolizer.data import LDrawReference
from nelegolizer import const
from typing import Tuple

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
        self.id = id
        self.dat_filename = id + ".dat"
        if self.dat_filename not in part_by_filename.keys():
            available_keys = list(part_by_filename.keys())
            raise KeyError(f"LegoBrick filename can be: {available_keys}. "
                           f"Filename {self.dat_filename} is invalid.")
        self.part = part_by_filename[self.dat_filename]
        self.mesh_position = mesh_position
        if rotation not in [0, 90, 180, 270]:
            raise Exception(f"LegoBrick rotation can be either 0, 90, 180"
                            f" or 270. Rotation {rotation} is invalid.")
        self.rotation = rotation
        self.color = color

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
        return cls(id=part_by_filename[ref.name].brick_id,
                   mesh_position=ref.position,
                   rotation=degrees,
                   color=ref.color)

    @property
    def matrix(self):
        rotation = np.zeros([4, 4])
        rotation[-1, :3] = self.mesh_position
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
