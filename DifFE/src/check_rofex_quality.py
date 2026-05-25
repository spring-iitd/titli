"""
Checks 1-3: Validate RoFex approximation quality BEFORE training NIDS models.

Check 1 — Output scale match: RoFex std / normalized AfterImage std (should be ~1.0)
Check 2 — Per-feature R²: how well does RoFex reproduce AfterImage structure
Check 3 — Anomaly signal direction: AUC of raw rofex features for benign vs malicious
           (> 0.7 → signal present, < 0.5 → signal inverted, ~0.5 → absent)

Usage:
    python check_rofex_quality.py --dataset kitsune
    python check_rofex_quality.py --dataset x-iot
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


def load(path, nrows=10000):
    df = pd.read_csv(path, nrows=nrows)
    df = df.drop(columns=[c for c in df.columns if str(c).lower() == 'label'], errors='ignore')
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)


def minmax_norm(df):
    mn, mx = df.min(), df.max()
    denom = (mx - mn).replace(0, 1)
    return 2 * (df - mn) / denom - 1


def check(dataset, data_root):
    data_root = Path(data_root)
    rofex_root = data_root / dataset / "features" / "rofex"
    ai_root    = data_root / dataset / "features" / "afterimage"

    attacks_b = sorted(p.stem for p in (rofex_root / "benign").glob("*.csv"))
    attacks_m = sorted(p.stem for p in (rofex_root / "malicious").glob("*.csv"))

    print(f"\n{'='*65}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*65}")

    # ── Check 1 & 2: scale match + R² per attack ──────────────────────
    print(f"\n{'─'*65}")
    print(f"  Check 1 — Output scale (RoFex std / AfterImage-normed std)")
    print(f"  Check 2 — Mean per-feature R² (RoFex ↔ AfterImage)")
    print(f"{'─'*65}")
    print(f"  {'Attack':25s}  {'Scale':6s}  {'Mean R²':8s}  {'Verdict'}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*8}  {'─'*15}")

    for attack in attacks_m[:8]:  # check up to 8 attacks
        rp = rofex_root / "malicious" / f"{attack}.csv"
        ap = ai_root    / "malicious" / f"{attack}.csv"
        if not rp.exists() or not ap.exists():
            continue
        rx = load(rp)
        ai = load(ap, nrows=len(rx))
        ai_n = minmax_norm(ai)

        n = min(len(rx), len(ai_n))
        rx, ai_n = rx.iloc[:n].values, ai_n.iloc[:n].values

        scale = rx.std() / (ai_n.std() + 1e-8)

        r2_vals = []
        for i in range(rx.shape[1]):
            ss_res = np.sum((ai_n[:, i] - rx[:, i]) ** 2)
            ss_tot = np.sum((ai_n[:, i] - ai_n[:, i].mean()) ** 2) + 1e-8
            r2_vals.append(1 - ss_res / ss_tot)
        mean_r2 = np.mean(r2_vals)

        scale_ok = "✓" if 0.3 < scale < 3.0 else "✗"
        r2_ok    = "✓" if mean_r2 > 0.3 else ("~" if mean_r2 > 0.1 else "✗")
        verdict  = "GOOD" if scale_ok == "✓" and r2_ok == "✓" else ("POOR" if r2_ok == "✗" else "MARGINAL")
        print(f"  {attack:25s}  {scale:5.2f}{scale_ok}  {mean_r2:8.4f}{r2_ok}  {verdict}")

    # ── Check 3: anomaly signal direction ─────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Check 3 — Anomaly signal: AUC(rofex L2 dist from benign mean)")
    print(f"  AUC > 0.7 → signal present  |  ~0.5 → absent  |  < 0.5 → INVERTED")
    print(f"{'─'*65}")
    print(f"  {'Attack':25s}  {'AUC':6s}  {'Verdict'}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*15}")

    # Compute benign mean from all available benign rofex features
    benign_frames = [load(p, nrows=5000) for p in sorted((rofex_root / "benign").glob("*.csv"))]
    if not benign_frames:
        print("  No benign rofex features found.")
        return
    benign_all = pd.concat(benign_frames, ignore_index=True).values
    benign_mean = benign_all.mean(axis=0)

    for attack in attacks_m:
        rp = rofex_root / "malicious" / f"{attack}.csv"
        lp = data_root / dataset / "labels" / "malicious" / f"{attack}.csv"
        if not rp.exists() or not lp.exists():
            continue

        rx = load(rp, nrows=50000).values
        lbl_df = pd.read_csv(lp, nrows=50000)
        labels = lbl_df.iloc[:, 1].values  # label is column index 1

        n = min(len(rx), len(labels))
        rx, labels = rx[:n], labels[:n]

        # L2 distance of each sample from benign mean → anomaly score
        scores = np.linalg.norm(rx - benign_mean, axis=1)

        if labels.sum() == 0 or labels.sum() == len(labels):
            print(f"  {attack:25s}  {'N/A':6s}  no label mix")
            continue
        try:
            auc = roc_auc_score(labels, scores)
        except Exception as e:
            print(f"  {attack:25s}  ERROR: {e}")
            continue

        if auc > 0.7:
            verdict = "SIGNAL PRESENT"
        elif auc < 0.4:
            verdict = "⚠ INVERTED"
        else:
            verdict = "~ WEAK/ABSENT"
        print(f"  {attack:25s}  {auc:.4f}  {verdict}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["kitsune", "x-iot"])
    parser.add_argument("--data-root", default=str(_DEFAULT_DATA_ROOT))
    args = parser.parse_args()
    check(args.dataset, args.data_root)


if __name__ == "__main__":
    main()
