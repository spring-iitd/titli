from __future__ import annotations

import importlib
from types import ModuleType
from typing import Tuple, Type


def _try_import(module: str) -> ModuleType | None:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def import_ids_module() -> ModuleType:
    mod = _try_import("titli.ids")
    if mod is not None:
        return mod
    mod = _try_import("titli.titli.ids")
    if mod is not None:
        return mod
    raise ModuleNotFoundError(
        "Could not import titli IDS module. Tried 'titli.ids' and 'titli.titli.ids'."
    )


def import_utils_module() -> ModuleType:
    mod = _try_import("titli.utils")
    if mod is not None:
        return mod
    mod = _try_import("titli.titli.utils")
    if mod is not None:
        return mod
    raise ModuleNotFoundError(
        "Could not import titli utils module. Tried 'titli.utils' and 'titli.titli.utils'."
    )


def import_fe_module() -> ModuleType:
    mod = _try_import("titli.fe")
    if mod is not None:
        return mod
    mod = _try_import("titli.titli.fe")
    if mod is not None:
        return mod
    raise ModuleNotFoundError(
        "Could not import titli feature extractor module. Tried 'titli.fe' and 'titli.titli.fe'."
    )


def get_streaming_csv_dataset_cls():
    utils = import_utils_module()
    if hasattr(utils, "StreamingCSVDataset"):
        return utils.StreamingCSVDataset
    # Some layouts expose it under utils.datasets
    datasets = _try_import(utils.__name__ + ".datasets")
    if datasets is not None and hasattr(datasets, "StreamingCSVDataset"):
        return datasets.StreamingCSVDataset
    raise AttributeError("Could not find StreamingCSVDataset in titli utils.")


def get_ids_class_map():
    ids = import_ids_module()
    mapping = {}
    for name in [
        "KitNET",
        "TorchKitNET",
        "PyTorchKitsune",
        "KitsuneIDS",
        "LOF",
        "OCSVM",
        "Autoencoder",
        "VAE",
        "ICL",
    ]:
        if hasattr(ids, name):
            mapping[name.lower()] = getattr(ids, name)
    return mapping

