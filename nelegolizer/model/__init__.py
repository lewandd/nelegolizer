"""
Load models from models/ to python dictionary 'models'
"""

from importlib.machinery import SourceFileLoader
from nelegolizer import path

models = SourceFileLoader("modules", path.BRICK_MODULES_FILE).load_module().load_all_models()