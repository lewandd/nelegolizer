import numpy as np

class LDrawReference():
    def __init__(self, *, 
                 name: str,
                 matrix: np.ndarray,
                 color: int):
        self.color = color
        self.matrix = matrix
        self.name = name

    @classmethod
    def from_line(cls, line: str):
        line = line.split(' ')
        line = [i.strip() for i in line]
        _, col, x, y, z, a, b, c, d, e, f, g, h, i = line[:14]
        color = int(col)
        matrix = np.array([
            [a, d, g, 0],
            [b, e, h, 0],
            [c, f, i, 0],
            [x, y, z, 1]
        ]).astype(float)
        name = " ".join(line[14:])
        return cls(name=name, matrix=matrix, color=color)

    @property
    def line(self) -> str:
        [a, d, g, _], [b, e, h, _], [c, f, i, _], [x, y, z, _] = self.matrix
        return f"1 {self.color} {x} {y} {z} {a} {b} {c} {d} {e} {f} {g} {h} {i} {self.name}\n"

    @property
    def position(self) -> np.ndarray:
        return self.matrix[3, :3]

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]