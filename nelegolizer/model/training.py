from .evaluation import evaluate_model
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Pełna deterministyczność w PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=16, lr=1e-3, device=None, seed=42):
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
        seed (int): ziarno generatora
    """
    set_seed(seed)

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