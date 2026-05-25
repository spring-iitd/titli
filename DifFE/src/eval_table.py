#!/usr/bin/env python
"""Print metric tables across (extractor, model) pairs for a dataset.

Reads per-sample dumps (artifacts/<dataset>/<ext>/<model>/<split>/<attack>.npz
written by score_dump_patch) and prints one ASCII table per metric, with
vanilla/defence rows paired and a Δ (vanilla − defence) column.

Metrics:
    auc                AUC-ROC
    eer                Equal Error Rate
    accuracy           Accuracy at the threshold saved during eval
    recall             TPR at the saved threshold (= attack-detection rate)
    tpr@fpr=X          TPR interpolated at FPR=X (X may be 0)
    fnr@fpr=X          1 − TPR@FPR=X

Examples:
    python eval_table.py --dataset x-iot --split malicious --metric auc
    python eval_table.py --dataset x-iot --split adversarial \\
        --metric "tpr@fpr=0.001" "fnr@fpr=0.001"
    python eval_table.py --dataset x-iot --split malicious \\
        --metric "tpr@fpr=0" --models tae kitnet
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

# cli key → (display name, dir name on disk)
MODEL_DIRS = {
    "tae":    ("TAE",    "transformer_ae"),
    "kitnet": ("KitNET", "kitnet"),
    "ae":     ("AE",     "autoencoder"),
    "if":     ("IF",     "isolation_forest"),
    "lof":    ("LOF",    "lof"),
}
DEFAULT_MODELS = ["tae", "kitnet", "ae", "if"]
DEFAULT_EXTRACTORS = ["afterimage", "rofex"]
DEFAULT_ARTIFACTS = "/home/kundan/DifFE/artifacts"

# preferred attack column order per dataset (overridable with --attacks)
DATASET_ATTACKS = {
    "x-iot":   ["SYN_DoS", "ACK_DoS", "UDP_DoS", "Port_Scanning", "Service_Detection"],
    "kitsune": ["Mirai", "SSDP_Flood", "SSL_Renegotiation", "OS_Scan", "SYN_DoS"],
}


# ── metrics ────────────────────────────────────────────────────────────────

def _roc(d):
    y = d["y_test"]
    if np.unique(y).size < 2:
        return None  # AUC/ROC undefined when only one class present
    return roc_curve(y, d["scores"], pos_label=1)


def _tpr_at(eps):
    def f(d):
        r = _roc(d)
        if r is None:
            return None
        fpr, tpr, _ = r
        if eps == 0:
            return float(tpr[fpr == 0].max()) if (fpr == 0).any() else 0.0
        return float(np.interp(eps, fpr, tpr))
    return f


def _fnr_at(eps):
    inner = _tpr_at(eps)
    return lambda d: None if inner(d) is None else 1.0 - inner(d)


def _auc(d):
    r = _roc(d)
    if r is None:
        return None
    fpr, tpr, _ = r
    return float(sk_auc(fpr, tpr))


def _eer(d):
    r = _roc(d)
    if r is None:
        return None
    fpr, tpr, _ = r
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def _accuracy(d):
    return float((d["y_pred"] == d["y_test"]).mean())


def _recall(d):
    pos = d["y_test"] == 1
    return float((d["y_pred"][pos] == 1).mean()) if pos.any() else None


def metric_fn(spec: str):
    s = spec.strip().lower()
    m = re.fullmatch(r"tpr@fpr=([\d.eE+-]+)", s)
    if m:
        eps = float(m.group(1))
        return _tpr_at(eps), f"TPR@FPR={eps:g}"
    m = re.fullmatch(r"fnr@fpr=([\d.eE+-]+)", s)
    if m:
        eps = float(m.group(1))
        return _fnr_at(eps), f"FNR@FPR={eps:g}"
    if s == "auc":      return _auc, "AUC"
    if s == "eer":      return _eer, "EER"
    if s == "accuracy": return _accuracy, "Accuracy"
    if s in {"recall", "tpr"}: return _recall, "Recall"
    raise ValueError(f"unknown metric: {spec!r}")


# ── data lookup ────────────────────────────────────────────────────────────

def npz_path(root: Path, dataset, ext, model_key, split, attack):
    return root / dataset / ext / MODEL_DIRS[model_key][1] / split / f"{attack}.npz"


def cell(root, dataset, ext, model_key, split, attack, fn):
    p = npz_path(root, dataset, ext, model_key, split, attack)
    if not p.exists():
        return None
    return fn(np.load(p))


def discover_attacks(root, dataset, split, extractors, models):
    sets = []
    for ext in extractors:
        for m in models:
            d = root / dataset / ext / MODEL_DIRS[m][1] / split
            if d.is_dir():
                sets.append({p.stem for p in d.glob("*.npz")})
    return sorted(set.intersection(*sets)) if sets else []


# ── rendering ──────────────────────────────────────────────────────────────

def render(root, dataset, split, label, fn, models, extractors, attacks):
    name_w = 22
    colw = max(8, max(len(a) for a in attacks))
    headers = attacks + ["Row_Avg", "Δ"]
    head = f"{'':<{name_w}}  " + "  ".join(f"{h:>{colw}}" for h in headers)

    print(f"\n=== {label}  |  dataset={dataset}  split={split} ===\n")
    print(head)
    print("─" * len(head))

    def fmt(v):
        return f"{v * 100:>{colw}.2f}" if v is not None else f"{'--':>{colw}}"

    for i, mk in enumerate(models):
        if i > 0:
            print("─" * len(head))
        # vanilla row (extractors[0])
        ext_v = extractors[0]
        v_vals = [cell(root, dataset, ext_v, mk, split, a, fn) for a in attacks]
        v_valid = [x for x in v_vals if x is not None]
        v_avg = (sum(v_valid) / len(v_valid)) if v_valid else None
        v_row = [fmt(x) for x in v_vals] + [fmt(v_avg), f"{'':>{colw}}"]
        print(f"{MODEL_DIRS[mk][0]:<{name_w}}  " + "  ".join(v_row))

        # defence row(s) — usually one
        for ext_d in extractors[1:]:
            d_vals = [cell(root, dataset, ext_d, mk, split, a, fn) for a in attacks]
            d_valid = [x for x in d_vals if x is not None]
            d_avg = (sum(d_valid) / len(d_valid)) if d_valid else None
            if v_avg is not None and d_avg is not None:
                drop = (v_avg - d_avg) * 100
                drop_cell = f"{drop:>+{colw}.2f}"
            else:
                drop_cell = f"{'--':>{colw}}"
            d_row = [fmt(x) for x in d_vals] + [fmt(d_avg), drop_cell]
            print(f"{'Defence-' + MODEL_DIRS[mk][0]:<{name_w}}  " + "  ".join(d_row))


# ── cli ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["malicious", "adversarial"])
    ap.add_argument("--metric", nargs="+", required=True,
                    help="auc | eer | accuracy | recall | tpr@fpr=X | fnr@fpr=X")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=list(MODEL_DIRS),
                    help=f"display order, vanilla→defence pair per model. default: {DEFAULT_MODELS}")
    ap.add_argument("--extractors", nargs="+", default=DEFAULT_EXTRACTORS,
                    help="vanilla first, defence second (default: afterimage rofex)")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--attacks", nargs="+", default=None,
                    help="explicit attack column order (default: dataset preset or alphabetical)")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    attacks = args.attacks
    if attacks is None:
        attacks = DATASET_ATTACKS.get(args.dataset)
    if attacks is None:
        attacks = discover_attacks(root, args.dataset, args.split, args.extractors, args.models)
    if not attacks:
        sys.exit(f"No attacks found under {root}/{args.dataset}/<ext>/<model>/{args.split}/*.npz")

    for spec in args.metric:
        fn, label = metric_fn(spec)
        render(root, args.dataset, args.split, label, fn, args.models, args.extractors, attacks)


if __name__ == "__main__":
    main()
