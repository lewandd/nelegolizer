"""
Models functions for object classification
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from . import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False

#MODELS_PATH = os.path.join(PACKAGE_PATH, "models/")

def test_predict(model, input):
    group_float = list(map(float, input))
    X = torch.tensor([group_float])#.to(device)
    pred = model(X)
    return pred.argmax(1).item()