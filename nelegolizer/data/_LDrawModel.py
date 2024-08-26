import numpy as np

from nelegolizer.data import LDrawReference, LegoBrick

class LDrawModel():
  def __init__(self, name: str):
    self.Name = name
    self.comms = {} # comments and commands
    self.references = []
  
  @classmethod
  def merge_multiple_models(cls, models):
    # copy informations from first model to new model
    merged = cls(models[0].Name)
    merged.comms = models[0].comms.copy()
    merged.references = models[0].references.copy()

    last_merged = cls(models[0].Name)
    model_by_name = {m.Name: m for m in models}

    # go len(my_dict) times to reach maximum depth possible
    for i in range(len(model_by_name)):
      last_merged.references = merged.references.copy()
      merged.references = []

      for ref in last_merged.references:
        if ref.name in model_by_name: # is model
          for ref_to_paste in model_by_name[ref.name].references:
            new_ref = LDrawReference(color=ref_to_paste.color,
                                    matrix=np.matmul(ref_to_paste.matrix, ref.matrix),
                                    name=ref_to_paste.name)
            merged.references.append(new_ref)
        else: # is single part
          merged.references.append(ref)
    return merged
  
  @classmethod
  def from_bricks(cls, name: str, bricks: list[LegoBrick]):
    model = cls(name)
    for b in bricks:
      
      model.references.append(LDrawReference(name=b.part.dat_filename, matrix=b.matrix, color=b.color))
    return model