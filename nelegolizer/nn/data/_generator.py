import numpy as np
import pandas as pd 
import nelegolizer.constants as CONST

def get_generator():
    return DataGenerator()

class DataGenerator():
    def __init__(self):
        self._idx = 0
        self._labels = []

    def generate(self, dir, subgenerator, label, num, prefix=""):
        for g in range(num):
            filename = f'{prefix}{self._idx + g}.txt'
            path = CONST.PATH + dir + filename
            self._labels.append((path, label))
            with open(path, "w") as f:
                df = pd.DataFrame([subgenerator()])
                f.write(df.to_csv(sep=str(' '), header=False, index=False))
        self._idx += num

    def create_label_file(self, dir, filename):
        path = CONST.PATH + dir + filename
        with open(path, "w") as f:
            df = pd.DataFrame(self._labels, columns=["Filename", "Label"])
            f.write(df.to_csv(sep=str(','), header=True, index=False))