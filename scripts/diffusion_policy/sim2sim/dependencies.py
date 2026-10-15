"""Explicit, optional MuJoCo dependency loading, without installation side effects."""
import importlib
from pathlib import Path
import sys


def load_mujoco(directory=None):
    if directory is not None:
        directory = Path(directory).resolve()
        if not (directory / "mujoco" / "__init__.py").is_file():
            raise FileNotFoundError(f"No isolated MuJoCo installation at {directory}")
        sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module("mujoco")
    except ImportError as error:
        raise ImportError("MuJoCo is optional; provide an explicit --dependency-dir installation.") from error
    if module.__version__ != "3.3.7":
        raise ValueError(f"Transfer contract pins MuJoCo 3.3.7, got {module.__version__}")
    return module
