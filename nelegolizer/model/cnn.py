# nelegolizer/models.py

import os
from typing import List, Tuple, Dict
import torch
from torch import nn
from nelegolizer import const, path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class V3DCNN1(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Blok 1: pełna rozdzielczość
        self.block1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Pooling 1 (łagodny, zmniejsza wymiar o połowę)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Blok 2: średnia skala
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Pooling 2 (mocniejszy, np. 3x3x3)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        # Blok 3: globalne cechy
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Klasyfikator
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.global_pool(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)

class Voxel3DCNN(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Konwolucje 3D
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # zmniejszamy wymiar

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Wyliczenie rozmiaru flattena
        with torch.no_grad():
            #if isinstance(input_shape, np.ndarray):
            #    print(f"got np.ndarray {type(input_shape)}")
            #    input_shape = tuple(int(x) for x in input_shape.tolist())
            #else:
            #    print(f"got tuple {input_shape}")
            #    input_shape = tuple(int(x) for x in input_shape)

            dummy = torch.zeros(1, 2, *input_shape)
            out = self.conv_layers(dummy)
            self.flattened_size = out.numel()

        # Klasyfikator
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

net_types = {"type3": V3DCNN1}


# Definicja: shape -> liczba klas
MODEL_CONFIGS: Dict[Tuple[int, int, int], int] = {
    (5, 2, 5): 2,
    (5, 6, 5): 7,
    (10, 6, 5): 3,
    (10, 6, 10): 3,
}


def get_model_shapes() -> List[Tuple[int, int, int]]:
    return list(MODEL_CONFIGS.keys())


def create_model(shape: Tuple[int, int, int]) -> nn.Module:
    """
    Tworzy model CNN dla danego kształtu wejścia (z paddingiem uwzględnionym).
    """
    if shape not in MODEL_CONFIGS:
        raise KeyError(f"No model with shape {shape}. Available: {MODEL_CONFIGS.keys()}")

    # Dodajemy padding
    #padded_shape = tuple(s + 2 * const.PADDING for s in shape)
    padded_shape = (shape[0] + 2*const.PADDING[0],
                    shape[1] + 2*const.PADDING[1],
                    shape[2] + 2*const.PADDING[2])
    num_classes = MODEL_CONFIGS[shape]

    return Voxel3DCNN(padded_shape, num_classes).to(device)

def test_predict_cnn(model, grid1, grid2, device=None):
    """
    Funkcja do predykcji etykiety dla pojedynczej pary gridów 3D.

    Args:
        model (torch.nn.Module): wytrenowany model 3D CNN
        grid1 (torch.Tensor lub np.ndarray): pierwszy grid 3D (D, H, W)
        grid2 (torch.Tensor lub np.ndarray): drugi grid 3D (D, H, W)
        device (str, opcjonalne): 'cuda' lub 'cpu'. Jeśli None -> automatyczne wykrycie.

    Returns:
        int: przewidziana etykieta klasy
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Konwersja do tensora
    if not isinstance(grid1, torch.Tensor):
        grid1 = torch.tensor(grid1, dtype=torch.float32)
    if not isinstance(grid2, torch.Tensor):
        grid2 = torch.tensor(grid2, dtype=torch.float32)

    # Stworzenie batcha o wymiarach [1, 2, D, H, W]
    x = torch.stack([grid1, grid2], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        predicted = outputs.argsort(dim=1, descending=True)[0].tolist()

    return predicted

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=16, lr=1e-3, device=None):
    """
    Funkcja do trenowania modelu 3D CNN na Twoim Dataset.

    Args:
        model: torch.nn.Module - model sieci neuronowej
        train_dataset: torch.utils.data.Dataset - dataset treningowy
        val_dataset: torch.utils.data.Dataset - dataset walidacyjny
        epochs (int): liczba epok
        batch_size (int): wielkość batcha
        lr (float): learning rate
        device (str): urządzenie ('cuda' lub 'cpu')
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataset)
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


def evaluate_model(model, dataloader, criterion=None, device=None):
    """
    Funkcja do ewaluacji modelu na zbiorze testowym/walidacyjnym.

    Args:
        model: torch.nn.Module - model do ewaluacji
        dataloader: torch.utils.data.DataLoader - dane do ewaluacji
        criterion: torch.nn.Module - funkcja straty (opcjonalnie)
        device (str): urządzenie ('cuda' lub 'cpu')

    Returns:
        avg_loss (float), accuracy (float)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            #if inputs.dim() == 4:
            #    inputs = inputs.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels)
            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if criterion is not None else 0.0
    accuracy = correct / total
    return avg_loss, accuracy