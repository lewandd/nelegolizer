"""
Print used device - either cpu or gpu.
"""
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device: {device}')