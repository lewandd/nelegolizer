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

@register_model("LegoNet2")
class LegoNet2(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, hidden_dim: int = 128): # 30 15 30
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Blok 1: pełna rozdzielczość
        self.block1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Pooling 1 (łagodny, zmniejsza wymiar o połowę)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 2, 3), stride=(3, 2, 3)) # teraz jest 10 7 10

        # Blok 2: średnia skala
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Pooling 2 (mocniejszy, np. 3x3x3)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)

        # Blok 3: globalne cechy
        self.block3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Klasyfikator
        self.fc_layers = nn.Sequential(
            nn.Linear(256, hidden_dim),
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


@register_model("LegoNet3")
class LegoNet3(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, hidden_dim: int = 128): # 30 15 30
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Blok 1
        self.block1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Blok 2
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(5, 2, 5), stride=(5, 2, 5))

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 3 * 3 * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("LegoNet4")
class LegoNet4(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, hidden_dim: int = 128): # 30 15 30
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5 * 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("PrimNet1")
class PrimNet1(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int): # 30 15 30
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 64

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    

@register_model("PrimNet2")
class PrimNet2(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 32

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    

@register_model("PrimNet3")
class PrimNet3(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 128

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("PrimNet4")
class PrimNet4(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int): # 30 15 30
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 64

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    

@register_model("PrimNet5")
class PrimNet5(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 32

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    

@register_model("PrimNet6")
class PrimNet6(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 128

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("PrimNet7")
class PrimNet7(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 64

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("PrimNet8")
class PrimNet8(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 32

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)
    
@register_model("PrimNet9")
class PrimNet9(nn.Module):
    """
    3D CNN do klasyfikacji voxelowych bloków LEGO.
    Przyjmuje dane o kształcie [batch, 2, D, H, W].

    Args:
        input_shape: (D, H, W) rozmiar wejścia (bez kanałów).
        num_classes: liczba klas do przewidzenia.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_shape = (30, 15, 30)
        self.num_classes = num_classes
        self.hidden_dim = 128

        self.block1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)  # [batch, features]
        return self.fc_layers(x)