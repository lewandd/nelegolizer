"""
Load models from models/ to python dictionary 'models'
"""

import torch
import os
from importlib.machinery import SourceFileLoader

PACKAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
MODELS_PATH = os.path.join(PACKAGE_PATH, "models/")

modules = SourceFileLoader("modules", os.path.join(PACKAGE_PATH, "scripts", "util", "modules.py")).load_module()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = {}

for name in modules.model_class:
    models[name] = modules.model_class[name]()
    
    # Load model
    MODEL_NAME = name + ".pth"
    MODEL_SAVE_PATH = MODELS_PATH + MODEL_NAME
    try:
        loaded = torch.load(f=MODEL_SAVE_PATH, map_location=device)
        models[name].load_state_dict(loaded)    
    except FileNotFoundError as e:
        print(f"No file {MODEL_SAVE_PATH}")
        continue
    else:
        print(f"Model succesfully loaded from: {MODEL_SAVE_PATH}")