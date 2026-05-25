"""
Compare AfterImage vs RoFex KitNET results from artifacts/.

Usage:
    python compare_results.py --dataset x-iot --split malicious
    python compare_results.py --dataset x-iot --split adversarial --metrics acc f1 auc
    python compare_results.py --dataset kitsune --split malicious --model kitnet
"""

import argparse
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_ARTIFACTS = _REPO_ROOT / "artifacts"

# Metric key → (label in file, display name, short column header)
METRIC_DEFS = {
    "acc":       ("Accuracy",    "Accuracy",  "Acc"),
    "precision": ("Precision",   "Precision", "Prec"),
    "recall":    ("Recall(TPR)", "Recall",    "Rec"),
    "f1":        ("F1 Score",    "F1",        "F1"),
    "auc":       ("AUC-ROC",     "AUC-ROC",  "AUC"),
    "eer":       ("EER",         "EER",       "EER"),
}
ALL_METRICS = list(METRIC_DEFS)


def parse_txt(path: Path) -> dict[str, float]:
    result = {}
    for line in path.read_text().splitlines():
        for key, (label, _, _) in METRIC_DEFS.items():
            if line.startswith(label + ":"):
                try:
                    result[key] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return result


def load_split(artifacts_root: Path, dataset: str, extractor: str,
               model: str, split: str) -> dict[str, dict]:
    d = artifacts_root / dataset / extractor / model / split
    if not d.exists():
        return {}
    return {p.stem: parse_txt(p) for p in sorted(d.glob("*.txt"))}


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  — "


def compare(dataset, split, model, metrics, artifacts_root):
    ai = load_split(artifacts_root, dataset, "afterimage", model, split)
    rx = load_split(artifacts_root, dataset, "rofex",      model, split)

    attacks = sorted(set(ai) | set(rx))
    if not attacks:
        print(f"No results found in {artifacts_root}/{dataset}/*/{ model}/{split}/")
        return

    col_w = 11  # width per extractor column
    hdrs = [METRIC_DEFS[m][2] for m in metrics]

    # Header
    print(f"\nDataset: {dataset}  |  Split: {split}  |  Model: {model}")
    print(f"Extractors: AfterImage (AI) vs RoFex (RX)\n")

    metric_header = "".join(f"  {'AI':>{col_w//2}} {'RX':<{col_w//2}}" for _ in metrics)
    sep = "-" * (22 + len(metrics) * (col_w + 2))
    print(f"{'Attack':<22}" + "".join(f"  {h:^{col_w}}" for h in hdrs))
    print(sep)

    for attack in attacks:
        a = ai.get(attack, {})
        r = rx.get(attack, {})
        row = f"{attack:<22}"
        for m in metrics:
            av, rv = a.get(m), r.get(m)
            row += f"  {fmt(av)} {fmt(rv)}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       required=True, choices=["kitsune", "x-iot"])
    parser.add_argument("--split",         required=True, choices=["malicious", "adversarial"])
    parser.add_argument("--model",         default="kitnet")
    parser.add_argument("--metrics",       nargs="+", default=ALL_METRICS,
                        choices=ALL_METRICS,
                        metavar="METRIC",
                        help=f"Any of: {ALL_METRICS} (default: all)")
    parser.add_argument("--artifacts-root", default=str(_DEFAULT_ARTIFACTS))
    args = parser.parse_args()

    compare(args.dataset, args.split, args.model, args.metrics,
            Path(args.artifacts_root))


if __name__ == "__main__":
    main()
