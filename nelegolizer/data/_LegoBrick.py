import pandas as pd
from nelegolizer.data import part_by_label

class LegoBrick:
    def __init__(self, *,
                 label: int, 
                 position: tuple[int, int, int], 
                 rotation: int):
        if label not in part_by_label.keys():
            raise KeyError(f"LegoBrick label can be: {list(part_by_label.keys())}. Label {label} is invalid.")
        self.label = label
        self.part = part_by_label[label]
        self.position = position
        if rotation not in [0, 90, 180, 270]:
            raise Exception(f"LegoBrick rotation can be either 0, 90, 180 or 270. Rotation {rotation} is invalid.")
        self.rotation = rotation

    def __str__(self):
        string = "LegoBrick: "
        string += "dat_filename=" + str(self.part.dat_filename) + ", "
        string += "label=" + str(self.label) + ", "
        string += "position=" + str(self.position) + ", "
        string += "rotation=" + str(self.rotation)
        return string 