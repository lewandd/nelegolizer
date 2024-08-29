"""
Load models from models/ to python dictionary 'models'
"""

from importlib.machinery import SourceFileLoader
import importlib.util
from nelegolizer import path

bc_models_loader = SourceFileLoader("bc_models", path.BRICK_MODULES_FILE)
bc_models_spec = importlib.util.spec_from_loader(name=bc_models_loader.name,
                                                 location=bc_models_loader)
bc_models_module = importlib.util.module_from_spec(bc_models_spec)
bc_models_loader.create_module(bc_models_spec)
bc_models_loader.exec_module(bc_models_module)
brick_classification_models = bc_models_module.load_all_models()
