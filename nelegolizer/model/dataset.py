import json
import torch
from torch.utils.data import Dataset
import numpy as np


class VoxelDataset(Dataset):
    def __init__(self, txt_file: str, transform=None, dtype=torch.float32):
        """
        Dataset ładujący dane voxelowe zapisane linia-po-linii w pliku .txt (JSONL).
        
        Args:
            txt_file (str): Ścieżka do pliku .txt z datasetem
            transform (callable, optional): Funkcja/transformacja nakładana na dane
            dtype (torch.dtype): Typ tensora zwracanego do modelu
        """
        self.txt_file = txt_file
        self.transform = transform
        self.dtype = dtype

        # Wczytanie wszystkich linii do pamięci (można też leniwie, jeśli plik jest ogromny)
        with open(txt_file, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # Parsujemy jedną linię
        data = json.loads(self.lines[idx])

        # Odtwarzamy kształt
        shape = tuple(data["shape"])

        # Tworzymy tensory dla obu kanałów
        grid1 = torch.tensor(data["channel1"], dtype=self.dtype).reshape(shape)
        grid2 = torch.tensor(data["channel2"], dtype=self.dtype).reshape(shape)

        # Stackujemy kanały -> [channels, D, H, W]
        x = torch.stack([grid1, grid2], dim=0)

        # Label
        y = torch.tensor(data["label"], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y