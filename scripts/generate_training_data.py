import os
import pandas as pd 
import numpy as np
from util import generator_funcs as generator

# Constant for package location

PACKAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
GENERATED_DATA_PATH = os.path.join(PACKAGE_PATH, "data/generated/")

class DataGeneratorManager():
    def __init__(self):
        self._idx = 0
        self._labels = []

    def generate(self, dir, generator, label, num, prefix=""):
        for g in range(num):
            filename = f'{prefix}{self._idx + g}.txt'
            path = os.path.join(dir, filename)
            self._labels.append((path, label))
            with open(path, "w") as f:
                df = pd.DataFrame([generator()])
                f.write(df.to_csv(sep=str(' '), header=False, index=False))
        self._idx += num

    def create_label_file(self, dir, filename):
        path = os.path.join(dir, filename)
        with open(path, "w") as f:
            df = pd.DataFrame(self._labels, columns=["Filename", "Label"])
            f.write(df.to_csv(sep=str(','), header=True, index=False))

if __name__ == '__main__':
    # generate train data for n111 neural network
    dir_n111 = os.path.join(GENERATED_DATA_PATH, "n111")
    gen_manager_n111 = DataGeneratorManager()
    gen_manager_n111.generate(dir=dir_n111, generator=generator.n111_full, label=0, num=10)
    gen_manager_n111.generate(dir=dir_n111, generator=generator.n111_lower_half, label=1, num=5)
    gen_manager_n111.create_label_file(dir=GENERATED_DATA_PATH, filename="labels_111.csv")

