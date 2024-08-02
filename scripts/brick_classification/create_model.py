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
from importlib.machinery import SourceFileLoader
from util import path

bc_modules = SourceFileLoader("modules", path.BRICK_MODULES_FILE).load_module()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0 :
        print("Usage: python3 create_model.py [model name] [model2 name] ... or python3 create_model.py all")
        sys.exit()
    elif argc == 1 and sys.argv[1] == "all":
        args = bc_modules.get_model_names()
    else:
        args = sys.argv[1:]
    
    for arg in args:
        model = bc_modules.create_model(arg)
        bc_modules.save_model(model, arg)