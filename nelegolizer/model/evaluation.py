import torch

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