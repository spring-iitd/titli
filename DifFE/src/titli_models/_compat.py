from __future__ import annotations

from importlib import import_module


def _try(path: str):
    try:
        return import_module(path)
    except Exception:
        return None


def get_pytorch_model_base():
    mod = _try("titli.ids.base_ids")
    if mod is None:
        mod = _try("titli.titli.ids.base_ids")
    if mod is None or not hasattr(mod, "PyTorchModel"):
        raise ModuleNotFoundError("Could not import PyTorchModel from titli (tried titli.ids.base_ids and titli.titli.ids.base_ids).")
    return mod.PyTorchModel


def get_create_directories():
    mod = _try("titli.utils.data")
    if mod is None:
        mod = _try("titli.titli.utils.data")
    if mod is None or not hasattr(mod, "create_directories"):
        raise ModuleNotFoundError("Could not import create_directories from titli (tried titli.utils.data and titli.titli.utils.data).")
    return mod.create_directories

