import torch
from typing import List
from ..data import LegoBrick
import numpy as np
from ..utils import brick as utils_brick

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

def any_tuple_close(list1, list2, rtol=1e-5, atol=1e-3):
    arr1 = np.array(list1)  # shape [N, 3]
    arr2 = np.array(list2)  # shape [M, 3]

    # Rozszerzenie wymiarów -> różnice pomiędzy wszystkimi parami
    diff = np.abs(arr1[:, None, :] - arr2[None, :, :])

    # Sprawdzenie czy którakolwiek para jest bliska
    return np.any(np.all(diff <= atol + rtol * np.abs(arr2[None, :, :]), axis=-1))

def compute_stability_cost(bricks: List[LegoBrick]) -> int:
    perp = 0

    utils_brick.normalize_positions(bricks, (0, 0, 0))
    for b in bricks:
        for n in bricks:
            if b.part.id == "3004" and n.part.id == "3004":
                n_under = any_tuple_close(b.lower_positions, n.occupied_positions)
                n_above = any_tuple_close(b.upper_positions, n.occupied_positions)
                if  n_under or n_above:
                    if ((b.rotation in [0, 180] and n.rotation in [90, 270]) or # są prostopadłe
                        (b.rotation in [90, 270] and n.rotation in [0, 180])):
                        perp += 1
    return int(perp/2)

def compute_iou(grid1: np.ndarray, grid2: np.ndarray) -> float:
    intersection = np.logical_and(grid1, grid2).sum()
    union = np.logical_or(grid1, grid2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return round(intersection / union, 2)