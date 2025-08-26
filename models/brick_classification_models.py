from torch import nn
import os
import torch
import numpy as np
from importlib.machinery import SourceFileLoader
import importlib.util
from typing import List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

__PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
__PATHS_FILE = os.path.join(__PACKAGE_DIR, "paths.py")
__CONSTANTS_FILE = os.path.join(__PACKAGE_DIR, "constants.py")

path_loader = SourceFileLoader("paths", __PATHS_FILE)
path_spec = importlib.util.spec_from_loader(path_loader.name, path_loader)
path = importlib.util.module_from_spec(path_spec)
path_loader.create_module(path_spec)
path_loader.exec_module(path)

const_loader = SourceFileLoader("constants", __CONSTANTS_FILE)
const_spec = importlib.util.spec_from_loader(const_loader.name, const_loader)
const = importlib.util.module_from_spec(const_spec)
const_loader.create_module(const_spec)
const_loader.exec_module(const)


class Model_n111(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        brick_shape = np.array([1, 1, 1])
        input_res = (brick_shape * const.BRICK_UNIT_RESOLUTION
                     + 2*const.PADDING)
        input_size = input_res[0] * input_res[1] * input_res[2]
        self.linear_relu_stack = nn.Sequential(
            # nn.Conv1d(60, 32, 8, stride=8),
            nn.Linear(input_size, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

modules = {
    (1, 1, 1): Model_n111
}

def get_model_shapes() -> List[Tuple[int]]:
    return list(modules.keys())

def create_model(shape: Tuple[int]) -> nn.Module:
    try:
        model = modules[shape]().to(device)
    except KeyError:
        raise KeyError(f"nn_modules: no model with shape {shape}. "
              f"Available models: {modules.keys()}")
    return model

def load_model(shape: Tuple[int], debug: bool = False) -> nn.Module:
    model = create_model(shape)
    MODEL_FILENAME = "model_n" + "".join(map(str, shape)) + ".pth"
    MODEL_PTH_FILE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_FILENAME)
    try:
        loaded = torch.load(f=MODEL_PTH_FILE_PATH, map_location=device)
        model.load_state_dict(loaded)
    except FileNotFoundError:
        print(f"FileNotFoundError: No file {MODEL_PTH_FILE_PATH}")
    else:
        if debug:
            print(f"Model succesfully loaded from: {MODEL_PTH_FILE_PATH}")
    return model
    

def load_shape_model_map():
    return {shape: load_model(shape) for shape in modules.keys()}

def save_model(model: nn.Module, shape: Tuple[int], debug: bool = False) -> None:
    MODEL_FILENAME = "model_n" + "".join(map(str, shape)) + ".pth"
    MODEL_PTH_FILE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_FILENAME)
    try:
        torch.save(obj=model.state_dict(), f=MODEL_PTH_FILE_PATH)
    except Exception as e:
        print(f"Exception: Exception occured while saving "
              f"model to {MODEL_PTH_FILE_PATH}: {e}")
    else:
        if debug:
            print(f"Model {MODEL_FILENAME} succesfully saved to: {MODEL_PTH_FILE_PATH}")