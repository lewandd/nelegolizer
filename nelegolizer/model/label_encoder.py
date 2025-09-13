class LabelEncoder:
    """
    Dwukierunkowy encoder mapujący (brick_id, rot) <-> int label.
    Obsługuje też klasę 'None' (brak klocka).
    """
    def __init__(self, label_map: dict):
        """
        Args:
            label_map (dict): np.
                {
                  ("None", 0): 0,
                  ("3004", 0): 1,
                  ("3004", 90): 2,
                  ...
                }
        """
        self.pair2idx = label_map
        self.idx2pair = {v: k for k, v in label_map.items()}

    def encode(self, pair):
        """(brick_id, rot) -> int"""
        return self.pair2idx[pair]

    def decode(self, idx):
        """int -> (brick_id, rot)"""
        return self.idx2pair[idx]

    def num_classes(self):
        return len(self.pair2idx)
    
def build_label_encoder(config):
    labels = config["model"]["labels"]
    mapping = {}
    idx = 0
    for brick_id, rotations in labels.items():
        for rot, class_id in rotations.items():
            mapping[(brick_id, int(rot))] = class_id
    return LabelEncoder(mapping)