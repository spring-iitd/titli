#!/usr/bin/env python
"""Chop merged kitsune-raw features + labels into benign/malicious/adversarial slices.

For each attack:
  cut = warmup length (1M, or 121,621 for Mirai)
  L   = min(mal_total - cut, adv_total - cut)   # adversarial-matched malicious slice
  features/benign/<A>.csv             = mal_feat[0:cut]
  features/malicious/<A>.csv          = mal_feat[cut:cut+L]
  features/adversarial/ghosturb/<A>.csv = adv_feat[cut:cut+L]
  labels/afterimage/{benign,malicious,adversarial/ghosturb}/<A>.csv = same indices
  (labels merged from data/kitsune/labels/{benign,malicious}/<A>.csv first.)
"""
import shutil
import sys
from pathlib import Path

R = Path("/home/kundan/DifFE/data/kitsune-raw")
D = Path("/home/kundan/DifFE/data/kitsune")

ATTACKS = {
    "Fuzzing":    1_000_000,
    "Mirai":        121_621,
    "SSDP_Flood": 1_000_000,
}


def line_count(p: Path) -> int:
    """Number of data rows (excluding header)."""
    with open(p) as f:
        return sum(1 for _ in f) - 1


def write_slice(src: Path, dst: Path, start: int, n: int) -> int:
    """Copy header + rows [start, start+n) from src to dst. Returns rows written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src) as fi, open(dst, "w") as fo:
        fo.write(fi.readline())  # header
        for _ in range(start):
            if not fi.readline():
                break
        for _ in range(n):
            ln = fi.readline()
            if not ln:
                break
            fo.write(ln)
            written += 1
    return written


def merge_labels(benign_lbl: Path, mal_lbl: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(benign_lbl) as fb, open(mal_lbl) as fm, open(dst, "w") as fo:
        fo.write(fb.readline())          # keep benign header
        for ln in fb:
            fo.write(ln)
        fm.readline()                    # skip malicious header
        for ln in fm:
            fo.write(ln)


def main():
    summary = []
    for attack, cut in ATTACKS.items():
        mal_feat_in = R / "features" / "afterimage" / "malicious" / f"{attack}.csv"
        adv_feat_in = R / "features" / "afterimage" / "adversarial" / "ghosturb" / f"{attack}.csv"

        mal_n = line_count(mal_feat_in)
        adv_n = line_count(adv_feat_in)
        L = min(mal_n - cut, adv_n - cut)
        if L <= 0:
            print(f"[!] {attack}: not enough rows past cut (mal={mal_n}, adv={adv_n}, cut={cut}); skipping")
            continue
        print(f"{attack}: cut={cut:,} mal_total={mal_n:,} adv_total={adv_n:,} → slice L={L:,}")

        # ── features ────────────────────────────────────────────────────────
        ben_out  = R / "features" / "afterimage" / "benign" / f"{attack}.csv"
        mal_tmp  = mal_feat_in.with_suffix(".csv.tmp")
        adv_tmp  = adv_feat_in.with_suffix(".csv.tmp")

        n_ben = write_slice(mal_feat_in, ben_out, 0, cut)
        n_mal = write_slice(mal_feat_in, mal_tmp, cut, L)
        n_adv = write_slice(adv_feat_in, adv_tmp, cut, L)
        mal_tmp.replace(mal_feat_in)
        adv_tmp.replace(adv_feat_in)

        # ── labels: merge → chop ────────────────────────────────────────────
        merged_lbl = R / "labels" /f"_merged_{attack}.csv"
        merge_labels(D / "labels" / "benign" / f"{attack}.csv",
                     D / "labels" / "malicious" / f"{attack}.csv",
                     merged_lbl)
        ben_lbl = R / "labels" /"benign"     / f"{attack}.csv"
        mal_lbl = R / "labels" /"malicious"  / f"{attack}.csv"
        adv_lbl = R / "labels" /"adversarial"/ "ghosturb" / f"{attack}.csv"
        nlb = write_slice(merged_lbl, ben_lbl, 0, cut)
        nlm = write_slice(merged_lbl, mal_lbl, cut, L)
        nla = write_slice(merged_lbl, adv_lbl, cut, L)
        merged_lbl.unlink()

        summary.append((attack, n_ben, n_mal, n_adv, nlb, nlm, nla))

    print("\n=== SUMMARY (rows written) ===")
    print(f"{'Attack':<14} {'feat_ben':>10} {'feat_mal':>10} {'feat_adv':>10} {'lbl_ben':>10} {'lbl_mal':>10} {'lbl_adv':>10}")
    for row in summary:
        attack, *vals = row
        print(f"{attack:<14} " + " ".join(f"{v:>10,}" for v in vals))


if __name__ == "__main__":
    main()
