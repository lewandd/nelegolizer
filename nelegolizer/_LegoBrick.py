import pandas as pd
from nelegolizer import PATH

_BRICKS_DATA_FILE = "/nelegolizer/bricks_data.csv"
_DF = pd.read_csv(PATH + _BRICKS_DATA_FILE)

class LegoBrick:
    """Represents Lego brick with all informations

    Attributes:
        _label (int) : label of brick used by neutral networks
        _position ((int, int, int)) : brick location x, y, z coordinates
        _rotation (int) : rotation around Y axis (could be 0, 90, 180 or 279)
    """

    def __init__(self, label, position, rotation):
        """Initialize attributes and create dictionary from csv data
        
        Attributes:
            label (int) : label of brick used by neutral networks
            position ((int, int, int)) : brick location x, y, z coordinates
            rotation (int) : rotation around Y axis (could be 0, 90, 180 or 279)
        """
        self._label = label
        self._position = position
        self._rotation = rotation

    def get_label(self):
        """Get label of brick used by neutral networks
        
        Returns:
            int : label
        """
        return self._label

    def get_position(self):
        """Get brick location of brick
        
        Returns:
            (int, int, int) : x, y, z coordinates
        """
        return self._position
    
    def get_rotation(self):
        """Get rotation of brick around Y axis
        
        Returns:
            int : rotation (could be 0, 90, 180 or 270)
        """
        return self._rotation

    def get_attr(self, attr):
        """Get value for given additional information dictionary key
        
        Attributes:
            attr (string) : dictionary key

        Returns:
            string : dictionary value
        """
        return self.get_dict()[attr]

    def get_dict(self):
        """Get additional information dictionary

        Returns:
            dict : dictionary
        """
        data = _DF.loc[self._label].to_dict()
        del data["Label"] # Label don't need to be included to additional informations
        return data

    def __str__(self):
        string = "LegoBrick Object : "
        string += "Label=" + str(self._label) + ", "
        string += "Position=" + str(self._position) + ", "
        string += "Rotation=" + str(self._rotation)
        return string 