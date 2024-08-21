from nelegolizer.data import part_by_label
from nelegolizer.utils import mesh as umesh
from nelegolizer.utils import grid

class LegoBrick:
    def __init__(self, *,
                 label: int, 
                 mesh_position: tuple[int, int, int], 
                 rotation: int,
                 color: int = 16):
        if label not in part_by_label.keys():
            raise KeyError(f"LegoBrick label can be: {list(part_by_label.keys())}. Label {label} is invalid.")
        self.label = label
        self.part = part_by_label[label]
        self.mesh_position = mesh_position
        if rotation not in [0, 90, 180, 270]:
            raise Exception(f"LegoBrick rotation can be either 0, 90, 180 or 270. Rotation {rotation} is invalid.")
        self.rotation = rotation
        self.color = color

    @property
    def mesh(self):
        m = self.part.mesh
        m = m.rotate_y(angle=self.rotation, inplace=False) 
        m = umesh.translate_to_zero(m)
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
        string += "label=" + str(self.label) + ", "
        string += "position=" + str(self.mesh_position) + ", "
        string += "rotation=" + str(self.rotation)
        string += "color=" + str(self.color)
        return string 