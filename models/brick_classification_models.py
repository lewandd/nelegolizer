from torch import nn
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PACKAGE = os.path.dirname(os.path.dirname(__file__))
BRICK_MODELS_DIR = os.path.join(PACKAGE, "models/brick_classification")

class Model_n111(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Conv1d(60, 32, 8, stride=8),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
nn_modules = {
    "model_n111": Model_n111
}

def get_model_names() -> list[str]:
    return list(nn_modules.keys())

def create_model(name: str) -> nn.Module: 
    try:
        model = nn_modules[name]().to(device)
    except KeyError:
        print(f"KeyError: No model like {name}. Available models: {get_model_names()}")
    return model

def load_model(model: nn.Module, name: str, debug: bool = False) -> nn.Module:
    MODEL_FILENAME = name + ".pth"
    MODEL_PTH_FILE_PATH = os.path.join(BRICK_MODELS_DIR, MODEL_FILENAME)
    try:
        loaded = torch.load(f=MODEL_PTH_FILE_PATH, map_location=device)
        model.load_state_dict(loaded)    
    except FileNotFoundError as e:
        print(f"FileNotFoundError: No file {MODEL_PTH_FILE_PATH}")
    else:
        if debug:
            print(f"Model succesfully loaded from: {MODEL_PTH_FILE_PATH}")
    return model

def save_model(model: nn.Module, name: str, debug: bool = False) -> None:
    MODEL_FILENAME = name + ".pth"
    MODEL_PTH_FILE_PATH = os.path.join(BRICK_MODELS_DIR, MODEL_FILENAME)
    try:
        torch.save(obj=model.state_dict(), f=MODEL_PTH_FILE_PATH)
    except Exception as e:
        print(f"Exception: Exception occured while saving model to {MODEL_PTH_FILE_PATH}: {e}")
    else:
        if debug:
            print(f"Model {name} succesfully saved to: {MODEL_PTH_FILE_PATH}")

def load_all_models():
    models = {}
    for name in get_model_names():
        model = create_model(name)
        model = load_model(model, name)
        models[name] = model
    return models