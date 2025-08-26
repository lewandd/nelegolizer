from nelegolizer.model import shape_label_division_id_map, shape_label_part_id_map, division_by_id

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
            