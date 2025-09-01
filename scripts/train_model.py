from nelegolizer.data import LDrawFile, initilize_parts, BrickOccupancy, ObjectOccupancy
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nelegolizer.model.dataset import VoxelDataset
from nelegolizer.model.io import load_model_str, save_model_str
from nelegolizer.model.cnn import Voxel3DCNN, train_model, evaluate_model
from torch.utils.data import random_split

def main():
    parser = argparse.ArgumentParser(description="Script that takes a model " \
                                    "input file and dataset txt file.")
    parser.add_argument(
        "model",
        type=str,
        help="Path to the input model file"
    )
    parser.add_argument(
        "shape",
        type=str,
        help='Shape could be either 111, 131, 231, 232'
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the input dataset file"
    )
    
    args = parser.parse_args()
    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    shape = tuple(int(ch) for ch in args.shape)
    shape = (shape[0]*5, shape[1]*2, shape[2]*5)

    if not model_path.exists():
        print(f"File doesn't exist: {model_path}")
        return
    if not dataset_path.exists():
        print(f"File doesn't exist: {dataset_path}")
        return
    
    initilize_parts()

    dataset = VoxelDataset(dataset_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    
    model = load_model_str(model_path, shape)
    #create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)


    train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.3e-3)

    save_model_str(model_path, model)

    # Trening
    # dla 232 najlepsze lr=1.3e-3
    #iter = 1
    #lr_1_3_acc = []
    #for i in range(iter):
    #    model = create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)
    #    train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.3e-3)
    #    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #    _, val_acc = evaluate_model(model, val_loader)
    #    lr_1_3_acc.append(val_acc)

    #lr_1_4_acc = []
    #for i in range(iter):
    #    model = create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)
    #    train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.4e-3)
    #    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #    _, val_acc = evaluate_model(model, val_loader)
    #    lr_1_4_acc.append(val_acc)

    #print(f"1.3e-3: {lr_1_3_acc} (mean: {sum(lr_1_3_acc)/len(lr_1_3_acc)}), 1.4e-3: {lr_1_4_acc} (mean: {sum(lr_1_4_acc)/len(lr_1_4_acc)})")
    
    #print(f"Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()