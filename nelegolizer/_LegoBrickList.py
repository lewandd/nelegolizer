
class LegoBrickList:
    """Represents list of LegoBrick objects

    Atributes:
        data (list) : list of LegoBrick objects
    """

    def __init__(self, list=[]):
        """Initialize list
        
        Attributes:
            list (list) : LegoBrick list 
        """
        self._data = list

    def __add__(self, other):
        """Addition operator

        Returns:
            LegoBrickList : concatenation of lists
        """
        return LegoBrickList(self._data + other._data)
    
    def __len__(self):
        """len operator
        
        Returns:
            int : length of list
        """
        return len(self._data)

    def __str__(self):
        string = "LegoBrickList Object : \n"
        for d in self._data:
            string += " + " + str(d) + "\n"
        return string
    
    def into_list(self):
        """Retruns data as list

        Returns:
            (list) : list of LegoBric objects
        """
        return self._data