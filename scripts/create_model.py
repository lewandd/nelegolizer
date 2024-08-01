"""
Create (or recreate if exists) models in models/

Usage:
python3 create_models.py model_name_1 model_name2 ...
or
python3 create_models.py all
"""

import sys
import os
import torch
from util.modules import model_class
from util import path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0 :
        print("Usage: python3 create_model.py [model name] [model2 name] ... or python3 create_model.py all")
        sys.exit()
    elif argc == 1 and sys.argv[1] == "all":
        args = list(model_class.keys())
    else:
        args = sys.argv[1:]
    
    for arg in args:
        # Create model
        try:
            model = model_class[arg]().to(device)
        except KeyError:
            print(f"No model like {arg}. Available models: {list(model_class.keys())}")
            continue

        # Save model
        MODEL_FILENAME = arg + ".pth"
        MODEL_PTH_FILE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_FILENAME)
        try:
            torch.save(obj=model.state_dict(), f=MODEL_PTH_FILE_PATH)
        except Exception as e:
            print(f"Exception occured while saving model to {MODEL_PTH_FILE_PATH}: {e}")
            sys.exit()
        else:
            print(f"Model {arg} succesfully saved to: {MODEL_PTH_FILE_PATH}")