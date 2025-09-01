import os
from typing import List, Tuple, Dict
import torch
from torch import nn
from nelegolizer import const, path
from nelegolizer.model.cnn import create_model, device, MODEL_CONFIGS

def model_filename(shape: Tuple[int, int, int]) -> str:
    return f"model_cnn_{''.join(map(str, shape))}.pth"


def model_path(shape: Tuple[int, int, int]) -> str:
    return os.path.join(path.BRICK_MODELS_DIR, model_filename(shape))


def save_model(model: nn.Module, shape: Tuple[int, int, int], debug: bool = False) -> None:
    filepath = model_path(shape)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    if debug:
        print(f"Model saved to {filepath}")

def save_model_str(filepath: str, model: nn.Module, debug: bool = False) -> None:
    torch.save(model.state_dict(), filepath)
    if debug:
        print(f"Model saved to {filepath}")

def load_model(shape: Tuple[int, int, int], debug: bool = False) -> nn.Module:
    model = create_model(shape)
    filepath = model_path(shape)
    if not os.path.exists(filepath):
        print(f"Model file not found: {filepath}")
        return None
    model.load_state_dict(torch.load(filepath, map_location=device))
    if debug:
        print(f"Model loaded from {filepath}")
    return model

def load_model_str(filepath: str, shape: Tuple[int, int, int], debug: bool = False) -> nn.Module:
    model = create_model(shape)
    if not os.path.exists(filepath):
        print(f"Model file not found: {filepath}")
        return None
    model.load_state_dict(torch.load(filepath, map_location=device))
    if debug:
        print(f"Model loaded from {filepath}")
    return model


def load_shape_model_map() -> Dict[Tuple[int, int, int], nn.Module]:
    return {shape: load_model(shape) for shape in MODEL_CONFIGS}