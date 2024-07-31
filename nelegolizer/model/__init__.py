"""
Load models from models/ to python dictionary 'models'
"""

import torch
import os
from importlib.machinery import SourceFileLoader
from nelegolizer import path

modules = SourceFileLoader("modules", path.BRICK_MODULES_FILE).load_module()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = {}

for name in modules.model_class:
    models[name] = modules.model_class[name]()
    
    # Load model
    MODEL_NAME = "{}.pth".format(name)
    MODEL_SAVE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_NAME)
    try:
        loaded = torch.load(f=MODEL_SAVE_PATH, map_location=device)
        models[name].load_state_dict(loaded)    
    except FileNotFoundError as e:
        print(f"No file {MODEL_SAVE_PATH}")
        continue
    else:
        print(f"Model succesfully loaded from: {MODEL_SAVE_PATH}")