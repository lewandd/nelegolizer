from .registry import register_model
from typing import List, Tuple, Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@register_model("LegoNet")
class LegoNet(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

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
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
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

#net_types = {"type3": LegoNet}


# Definicja: shape -> liczba klas
#MODEL_CONFIGS: Dict[Tuple[int, int, int], int] = {
#    (5, 2, 5): 2,
#    (5, 6, 5): 7,
#    (10, 6, 5): 3,
#    (10, 6, 10): 3,
#}

#def create_model(shape: Tuple[int, int, int]) -> nn.Module:
#    """
#    Tworzy model CNN dla danego kształtu wejścia (z paddingiem uwzględnionym).
#    """
#    if shape not in MODEL_CONFIGS:
#        raise KeyError(f"No model with shape {shape}. Available: {MODEL_CONFIGS.keys()}")

    # Dodajemy padding
    #padded_shape = tuple(s for s in shape)
#    padded_shape = (shape[0],
#                    shape[1],
#                    shape[2])
#    num_classes = MODEL_CONFIGS[shape]

#    return Voxel3DCNN(padded_shape, num_classes).to(device)
