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
from util.modules import nn_modules, create_model, save_model
from util import path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0 :
        print("Usage: python3 create_model.py [model name] [model2 name] ... or python3 create_model.py all")
        sys.exit()
    elif argc == 1 and sys.argv[1] == "all":
        args = list(nn_modules.keys())
    else:
        args = sys.argv[1:]
    
    for arg in args:
        model = create_model(arg)
        save_model(model, arg)