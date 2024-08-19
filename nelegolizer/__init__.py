from importlib.machinery import SourceFileLoader
import importlib.util
import os

__PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
__PATHS_FILE = os.path.join(__PACKAGE_DIR, "paths.py")
__CONSTANTS_FILE = os.path.join(__PACKAGE_DIR, "constants.py")

path_loader = SourceFileLoader("paths", __PATHS_FILE)
path_spec = importlib.util.spec_from_loader(path_loader.name, path_loader)
path = importlib.util.module_from_spec(path_spec)
path_loader.create_module(path_spec)
path_loader.exec_module(path)

const_loader = SourceFileLoader("constants", __CONSTANTS_FILE)
const_spec = importlib.util.spec_from_loader(const_loader.name, const_loader)
const = importlib.util.module_from_spec(const_spec)
const_loader.create_module(const_spec)
const_loader.exec_module(const)