import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, grid1, grid2, device=None):
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