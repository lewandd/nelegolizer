class LDrawModel():
  def __init__(self, name: str):
    self.Name = name
    self.comms = {} # comments and commands
    self.references = []