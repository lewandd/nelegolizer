from nelegolizer.model import shape_label_division_id_map, shape_label_part_id_map, division_by_id
from nelegolizer.model import shape_label_part_id_map2, subshapes_by_id, shape_label_subshape_id_map2, shape_label_part_rot_map2

class ClassificationResult:
    def __init__(self, shape, label):
        self.shape = shape
        self.label = label
        self.type = None
        self.brick_id = None
        self.division_id = None
        self.division = None
        
        shape_t = tuple(map(int, shape))
        if (shape_t in shape_label_part_id_map.keys()) and (label in shape_label_part_id_map[shape_t].keys()):
            self.type = "brick"
            self.brick_id = shape_label_part_id_map[shape_t][label]
        elif (shape_t in shape_label_division_id_map.keys()) and (label in shape_label_division_id_map[shape_t].keys()):
            self.type = "division"
            self.division_id = shape_label_division_id_map[shape_t][label]
            self.division = division_by_id[self.division_id]
            
class ClassificationResult2:
    def __init__(self, shape, label):
        self.shape = shape
        self.label = label
        self.type = None
        self.brick_id = None
        self.rotation = None
        self.subshape = None
        
        shape_t = tuple(map(int, shape))
        if (shape_t in shape_label_part_id_map2.keys()) and (label in shape_label_part_id_map2[shape_t].keys()):
            self.type = "brick"
            self.rotation = shape_label_part_rot_map2[shape_t][label]
            self.brick_id = shape_label_part_id_map2[shape_t][label]
        elif (shape_t in shape_label_subshape_id_map2.keys()) and (label in shape_label_subshape_id_map2[shape_t].keys()):
            self.type = "subshape"
            self.subshape_id = shape_label_subshape_id_map2[shape_t][label]
            self.subshape = subshapes_by_id[self.subshape_id]

            # sieć zwraca w result nic więcej o
# tym subspace tylko docelowy realny
# rozmiar czyli np (2;1;1) czy (1;1;2)
# na tej podstawie wyboerany jest kolejny
# grid w głębszej rekurencji
# za to gdy na tym shape'ie
# ma być wyskonywana kolejna klasyfikacja
# to wtedy wybieran jest analogiczny
# obrócony shape dla sieci i wtedy też
# cały grid się obraca o 90 stopni
# jedynie na potrzeby klasyfikacji;
# jeżeli wybrano nowy podział to
# można zapomnieć o tej poprzedniej 
# rotacji bo to nieistotnepo prostu
# wybiera się kolejną podprzestrzeń

# jeżeli wybrany był klocek to
# uwzględniana jest ta rotacja
# w obębie tej jednej grupy może być
# albo 0 albo 90 i to ma być 
# uwzględnione w określeniu ostatecznej
# rotacji bo każdy label klocka ma mieć
# dopisane w wyniku jaka to rotacja

# w przypadku subspace pisanie o rotacji
# gdziekolwiek w danych jest nieistotne