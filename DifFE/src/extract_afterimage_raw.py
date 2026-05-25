#!/usr/bin/env python
"""Extract AfterImage features from a pcap → CSV.

State is built continuously from packet 0; --limit caps the *output* row count
but the underlying FE still processes packets sequentially.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# Add src/ to path so `afterimage` package is importable when run as a script.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many packets (0 = full pcap).")
    ap.add_argument("--progress-every", type=int, default=50000)
    args = ap.parse_args()

    from afterimage import FE  # imports scapy + does tshark conversion in __init__

    fe = FE(args.pcap, limit=args.limit if args.limit > 0 else float("inf"))
    headers = fe.nstat.getNetStatHeaders()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tag = Path(args.pcap).parent.name + "/" + Path(args.pcap).stem
    t0 = time.time()
    last_t = t0
    last_i = 0

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        i = 0
        while True:
            v = fe.get_next_vector()
            if len(v) == 0:
                break
            w.writerow(v)
            i += 1
            if i % args.progress_every == 0:
                now = time.time()
                rate = (i - last_i) / max(now - last_t, 1e-6)
                print(f"[{tag}] {i:,} pkts | {rate:,.0f} pps | elapsed {now-t0:.0f}s",
                      flush=True)
                last_t, last_i = now, i

    print(f"[{tag}] DONE: {i:,} pkts in {time.time()-t0:.0f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
