import sys

from pathlib import Path

import importlib


def add_repo_to_path():
    repo_root = Path(__file__).resolve().parents[2]

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_model_module(model_dir):

    add_repo_to_path()

    module = importlib.import_module(model_dir + ".utilities.model_loader")

    return module
