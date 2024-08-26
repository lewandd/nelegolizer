from nelegolizer.data import LDrawModel, LDrawReference

class LDrawFile():
    def __init__(self):
        self.path = None
        self.lines = []
        self.models = []

    @classmethod
    def load_from_file(cls, path: str):
        c = cls()
        c.path = path
        file = open(path, mode = 'r', encoding = 'utf-8-sig')
        c.lines = file.readlines()
        file.close()

        # load models
        act_model = None
        for line in c.lines:
            line_list = [i.strip() for i in line.split(' ')]
            line_head = line_list[:2]
            line_tail = " ".join(line_list[2:])

            match line_head:
                case ['0', 'Name:']:
                    name = line_tail
                    act_model = LDrawModel(name)
                    act_model.comms['Name'] = name
                    c.models.append(act_model)
                case ['0', comm]:
                    if act_model:
                        act_model.comms[comm.strip(":")] = line_tail
                case ['1', _]:
                    act_model.references.append(LDrawReference.from_line(line))
        return c
        
    def add_model_lines(self, model: LDrawModel):
        self.models.append(model)
        for comm in model.comms:
            if comm in ['Name', 'Author']:
                self.lines.append(f"0 {comm}: {model.comms[comm]}\n")
            else:
                self.lines.append(f"0 {comm} {model.Name}\n")

        for ref in model.references:
            self.lines.append(ref.line)