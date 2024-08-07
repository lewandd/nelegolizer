"""
Models functions for object classification
"""

import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False

def test_predict(model, input):
    group_float = list(map(float, input))
    X = torch.tensor([group_float])#.to(device)
    pred = model(X)
    return pred.argmax(1).item()