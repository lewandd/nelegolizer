from ._generator import get_generator
import nelegolizer.nn.data.subgenerator as subgen
import os

GENERATED_DATA_PATH = "/nelegolizer/nn/data/generated/"

def for111():
    generated_files_dir = GENERATED_DATA_PATH+"111/"
    
    gen = get_generator()
    gen.generate(dir=generated_files_dir, subgenerator=subgen._111_full, label=1, num=10)
    gen.generate(dir=generated_files_dir, subgenerator=subgen._111_lower_half, label=2, num=5)
    gen.create_label_file(dir=GENERATED_DATA_PATH, filename="labels_111.csv")

def all():
    for111()

if __name__ == '__main__':
    all()