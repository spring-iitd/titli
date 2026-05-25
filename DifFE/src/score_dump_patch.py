"""Side-effect: monkey-patch titli's PyTorchModel.evaluate to dump per-sample
(y_test, y_pred, scores, threshold) as .npz alongside the metrics .txt.

Captures enough to recompute any threshold-based metric post-hoc — TPR@FPR=ε,
FNR@FPR=ε, full ROC, PR curve, etc:

    d = np.load("…/<attack>.npz")
    fpr, tpr, thr = sklearn.metrics.roc_curve(d["y_test"], d["scores"], pos_label=1)
    tpr_at_fpr_001 = np.interp(0.001, fpr, tpr)
    fnr_at_fpr_001 = 1 - tpr_at_fpr_001

Dump path mirrors metrics: ./artifacts/{dataset_name}/objects/scores/{model_name}.npz
(relative to cwd, same as titli's metrics file). benchmark.py moves it to the
final results dir alongside the .txt.

All five registered IDS models (KitNET, Autoencoder, TransformerAE, IsolationForest,
LOF) funnel through PyTorchModel.evaluate, so a single patch covers everything.
"""
import os
import numpy as np
from titli.ids import base_ids

_PATCHED_FLAG = "_score_dump_patched"

if not getattr(base_ids.PyTorchModel, _PATCHED_FLAG, False):
    _ORIG_EVALUATE = base_ids.PyTorchModel.evaluate

    def _evaluate_with_dump(self, y_test, y_pred, reconstruction_errors):
        out_dir = f"./artifacts/{self.dataset_name}/objects/scores"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            f"{out_dir}/{self.model_name.lower()}.npz",
            y_test=np.asarray(y_test).ravel(),
            y_pred=np.asarray(y_pred).ravel(),
            scores=np.asarray(reconstruction_errors).ravel(),
            threshold=np.asarray(getattr(self, "threshold", np.nan), dtype=float),
        )
        return _ORIG_EVALUATE(self, y_test, y_pred, reconstruction_errors)

    base_ids.PyTorchModel.evaluate = _evaluate_with_dump
    setattr(base_ids.PyTorchModel, _PATCHED_FLAG, True)
