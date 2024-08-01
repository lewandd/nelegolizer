import os
import pandas as pd 
import numpy as np
from util import generator_funcs as generator
from util import path

class DataGeneratorManager():
    def __init__(self):
        self._idx = 0
        self._labels_data = []

    def generate(self, data_dir, generator, label, num, prefix=""):
        for _ in range(num):
            data_filename = f'{prefix}{self._idx}.txt'
            data_path = os.path.join(data_dir, data_filename)
            self._labels_data.append((data_filename, label))
            with open(data_path, "w") as f:
                data_df = pd.DataFrame([generator()])
                data_csv = data_df.to_csv(sep=str(' '), header=False, index=False)
                f.write(data_csv)
            self._idx += 1

    def create_label_file(self, path):
        with open(path, "w") as f:
            label_data_df = pd.DataFrame(self._labels_data, columns=["Filename", "Label"])
            label_data_csv = label_data_df.to_csv(sep=str(','), header=True, index=False)
            f.write(label_data_csv)

def generate_for_n111():
    MODEL_DATA_DIR = os.path.join(path.BRICK_CLASSFICATION_DATA_DIR, "n111")
    TRAIN_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "train_data_labels.csv")
    TEST_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "test_data_labels.csv")
    TRAIN_DATA_DIR = os.path.join(MODEL_DATA_DIR, "train_data")
    TEST_DATA_DIR = os.path.join(MODEL_DATA_DIR, "test_data")
    
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    train_gm = DataGeneratorManager()
    train_gm.generate(data_dir=TRAIN_DATA_DIR, generator=generator.n111_full, label=0, num=10, prefix="train-n111-0-")
    train_gm.generate(data_dir=TRAIN_DATA_DIR, generator=generator.n111_lower_half, label=1, num=5, prefix="train-n111-1-")
    train_gm.create_label_file(path=TRAIN_LABEL_FILE_PATH)

    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    test_gm = DataGeneratorManager()
    test_gm.generate(data_dir=TEST_DATA_DIR, generator=generator.n111_full, label=0, num=5, prefix="test-n111-0-")
    test_gm.generate(data_dir=TEST_DATA_DIR, generator=generator.n111_lower_half, label=1, num=2, prefix="test-n111-1-")
    test_gm.create_label_file(path=TEST_LABEL_FILE_PATH)

if __name__ == '__main__':
    generate_for_n111()


