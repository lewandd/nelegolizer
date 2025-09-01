from nelegolizer.data import LDrawFile, initilize_parts, BrickOccupancy, ObjectOccupancy
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nelegolizer.model.dataset import VoxelDataset
from nelegolizer.model.io import load_model_str, create_model, save_model_str
from nelegolizer.model.cnn import Voxel3DCNN, train_model, evaluate_model
from torch.utils.data import random_split

def main():
    parser = argparse.ArgumentParser(description="Script that takes a model " \
                                    "input file and dataset txt file.")
    parser.add_argument(
        "model",
        type=str,
        help="Path to the output model file"
    )
    parser.add_argument(
        "shape",
        type=str,
        help='Shape could be either 111, 131, 231, 232'
    )

    args = parser.parse_args()
    model_path = Path(args.model)
    
    shape = tuple(int(ch) for ch in args.shape)
    shape = (shape[0]*5, shape[1]*2, shape[2]*5)
    
    model = create_model(shape)
    save_model_str(model_path, model)

if __name__ == "__main__":
    main()