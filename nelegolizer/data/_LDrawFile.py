from . import LDrawModel, LDrawReference


class LDrawFile():
    def __init__(self):
        self.path = None
        self.lines = []
        self.models = []

    @classmethod
    def load(cls, path: str):
        c = cls()
        c.path = path
        file = open(path, mode='r', encoding='utf-8-sig')
        c.lines = file.readlines()
        file.close()

        # load models
        act_model = None
        for line in c.lines:
            line_list = [i.strip() for i in line.split(' ')]
            line_head = line_list[:2]
            line_tail = " ".join(line_list[2:])

            if line_head[0] == '0' and line_head[1] == 'Name:':
                name = line_tail
                act_model = LDrawModel(name)
                act_model.comms['Name'] = name
                c.models.append(act_model)
            elif line_head[0] == '0':
                comm = line_head[1]
                if act_model:
                    act_model.comms[comm.strip(":")] = line_tail
            elif line_head[0] == '1':
                act_model.references.append(LDrawReference.from_line(line))
        return c

    def save(self, path: str) -> None:
        self.path = path
        with open(path, 'w') as f:
            for line in self.lines:
                f.write(line)

    def add_model(self, model: LDrawModel):
        self.models.append(model)
        for comm in model.comms:
            if comm in ['Name', 'Author']:
                self.lines.append(f"0 {comm}: {model.comms[comm]}\n")
            else:
                self.lines.append(f"0 {comm} {model.Name}\n")

        for ref in model.references:
            self.lines.append(ref.line)
