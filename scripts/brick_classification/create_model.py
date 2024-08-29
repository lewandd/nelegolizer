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
import importlib.util

__PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
__PATHS_FILE = os.path.join(__PACKAGE_DIR, "paths.py")

path_loader = SourceFileLoader("paths", __PATHS_FILE)
path_spec = importlib.util.spec_from_loader(path_loader.name, path_loader)
path = importlib.util.module_from_spec(path_spec)
path_loader.create_module(path_spec)
path_loader.exec_module(path)

bc_models_loader = SourceFileLoader("bc_models", path.BRICK_MODULES_FILE)
bc_models_spec = importlib.util.spec_from_loader(bc_models_loader.name,
                                                 bc_models_loader)
bc_models_module = importlib.util.module_from_spec(bc_models_spec)
bc_models_loader.create_module(bc_models_spec)
bc_models_loader.exec_module(bc_models_module)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0:
        print("Usage: python3 create_model.py [model name] [model2 name] ... "
              "or python3 create_model.py all")
        sys.exit()
    elif argc == 1 and sys.argv[1] == "all":
        args = bc_models_module.get_model_names()
    else:
        args = sys.argv[1:]

    for arg in args:
        model = bc_models_module.create_model(arg)
        bc_models_module.save_model(model, arg)
