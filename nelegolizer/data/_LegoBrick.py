import numpy as np

from nelegolizer.data import part_by_label
from nelegolizer.utils import mesh as umesh

class LegoBrick:
    def __init__(self, *,
                 label: int, 
                 position: tuple[int, int, int], 
                 rotation: int,
                 color: int = 16):
        if label not in part_by_label.keys():
            raise KeyError(f"LegoBrick label can be: {list(part_by_label.keys())}. Label {label} is invalid.")
        self.label = label
        self.part = part_by_label[label]
        self.position = position
        if rotation not in [0, 90, 180, 270]:
            raise Exception(f"LegoBrick rotation can be either 0, 90, 180 or 270. Rotation {rotation} is invalid.")
        self.rotation = rotation
        self.color = color

    @property
    def mesh(self):
        m = self.part.mesh
        m = m.rotate_y(angle=self.rotation, inplace=False) 
        m = umesh.translate_to_zero(m)
        m = m.translate(np.array(self.position) * np.array([0.8, 1.12, 0.8]), inplace=False)
        return m

    def __str__(self):
        string = "LegoBrick: "
        string += "dat_filename=" + str(self.part.dat_filename) + ", "
        string += "label=" + str(self.label) + ", "
        string += "position=" + str(self.position) + ", "
        string += "rotation=" + str(self.rotation)
        string += "color=" + str(self.color)
        return string 