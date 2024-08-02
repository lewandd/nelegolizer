import torch
import pandas as pd
import os
from torch.utils.data import Dataset

class CustomVoxelsDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.data_labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_labels.iloc[idx, 0])
        f = open(data_path, "r")
        values = list(map(float, f.readline().split(" ")))
        tensor_data = torch.tensor(values)
        label = self.data_labels.iloc[idx, 1]
        if self.transform:
            tensor_data = self.transform(tensor_data)
        if self.target_transform:
            label = self.target_transform(label)
        return tensor_data, label