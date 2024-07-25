import pandas as pd
import nelegolizer.constants as CONST

_BRICKS_DESC_PATH = CONST.PATH + "/nelegolizer/data/brick_descriptions.csv"
_DF = pd.read_csv(_BRICKS_DESC_PATH)

class LegoBrick:
    def __init__(self, *,
                 label: int, 
                 position: tuple[int, int, int], 
                 rotation: int,
                 DF: pd.DataFrame = _DF):
        self.__label = label
        self.__position = position
        if rotation not in [0, 90, 180, 270]:
            raise Exception("LegoBrick rotation can be either 0, 90, 180 or 270")
        self.__rotation = rotation
        self.attribute = DF.loc[self.__label].to_dict()

    @property
    def label(self) -> int:
        return self.__label

    @property
    def position(self) -> tuple[int, int, int]:
        return self.__position
    
    @property
    def rotation(self) -> int:
        return self.__rotation

    def __str__(self):
        string = "LegoBrick Object : "
        string += "Label=" + str(self.__label) + ", "
        string += "Position=" + str(self.__position) + ", "
        string += "Rotation=" + str(self.__rotation)
        return string 