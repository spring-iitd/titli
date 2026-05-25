import argparse
import contextlib
import io
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import score_dump_patch  # noqa: F401 — installs PyTorchModel.evaluate score dumper
from titli.ids import KitNET
from titli_compat import get_streaming_csv_dataset_cls
from titli_models import (
    Autoencoder,
    TransformerAE,
    LOF,
    IsolationForest,
)

_SRC_DIR = Path(__file__).parent
_REPO_ROOT = _SRC_DIR.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
_DEFAULT_ARTIFACTS_ROOT = _REPO_ROOT / "artifacts"
_DEFAULT_CHECKPOINTS_ROOT = _REPO_ROOT / "checkpoints"
# titli writes its internal artifacts relative to cwd; we always chdir here before save/load/evaluate
_TITLI_SCRATCH = _SRC_DIR / "artifacts"

MODEL_REGISTRY = {
    "kitnet": KitNET,
    "autoencoder": Autoencoder,
    "transformer_ae": TransformerAE,
    "lof": LOF,
    "isolation_forest": IsolationForest,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate IDS models on extracted feature CSVs.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=["kitsune", "x-iot", "kitsune-raw"])
    parser.add_argument("--extractor", required=True, choices=["afterimage", "rofex", "rofex-transformer"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["malicious", "adversarial"],
                        choices=["malicious", "adversarial"])
    parser.add_argument("--adv-variant", default="ghosturb",
                        help="Adversarial subdirectory name (default: ghosturb)")
    parser.add_argument("--data-root", default=str(_DEFAULT_DATA_ROOT))
    parser.add_argument("--artifacts-root", default=str(_DEFAULT_ARTIFACTS_ROOT))
    parser.add_argument("--checkpoints-root", default=str(_DEFAULT_CHECKPOINTS_ROOT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=1000000)
    parser.add_argument("--max-test-samples", type=int, default=50000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--concurrent", action="store_true",
                        help="Evaluate all files in parallel (folder mode only).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for --concurrent (default: auto).")
    parser.add_argument("--model-name", default=None,
                        help="Optional suffix appended to model key, e.g. 'ssdp_flood' → "
                             "checkpoint stored as kitnet_ssdp_flood/.")
    parser.add_argument("--benign-file", default=None,
                        help="Benign feature filename (stem or full name) to use for training. "
                             "Required when multiple benign files exist.")
    parser.add_argument("--malicious-file", default=None,
                        help="Malicious/adversarial feature filename (stem or full name) to evaluate. "
                             "When absent, all files in the split directory are evaluated.")
    return parser.parse_args()


def build_model(model_key, dataset_key, input_size, device):
    cls = MODEL_REGISTRY[model_key]
    return cls(dataset_name=dataset_key, input_size=input_size, device=device)


def _model_key_full(args) -> str:
    """Returns model key with optional --model-name suffix, e.g. 'kitnet_ssdp_flood'."""
    return f"{args.model}_{args.model_name}" if args.model_name else args.model


def _checkpoint_models_dir(checkpoints_root: Path, dataset: str, extractor: str, model_key: str) -> Path:
    return checkpoints_root / dataset / extractor / model_key


def _stage_models(dst: Path, src: Path) -> None:
    """Copy all model files from src/ into dst/, creating dst if needed."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        if f.is_file():
            shutil.copy2(str(f), str(dst / f.name))


def _cleanup_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(str(path), ignore_errors=True)


BENIGN_SAMPLES_PER_FILE = 50000
_EMBEDDED_LABEL_COLS = {"Label", "label"}
# Label CSVs have ['Unnamed: 0', 'x']; the actual label is always at column index 1.
_LABEL_COLUMN = 1


def _strip_embedded_label(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if c in _EMBEDDED_LABEL_COLS]
    return df.drop(columns=to_drop) if to_drop else df


def _concat_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    return pd.concat(
        [_strip_embedded_label(pd.read_csv(p, nrows=BENIGN_SAMPLES_PER_FILE)) for p in sorted(csv_paths)],
        ignore_index=True,
    )


def _write_temp(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _get_input_size(csv_path: Path) -> int:
    return len(_strip_embedded_label(pd.read_csv(csv_path, nrows=0)).columns)



def _resolve_benign_csv(feat_dir: Path, lbl_dir: Path, benign_file: str | None,
                         tmp_dir: Path, key: str) -> tuple[Path, Path | None]:
    """Return (feat_csv, lbl_csv_or_None) for the benign training file."""
    feat_csvs = sorted(feat_dir.glob("*.csv"))
    if not feat_csvs:
        raise FileNotFoundError(f"No benign feature CSVs found in {feat_dir}")

    if len(feat_csvs) == 1:
        feat = feat_csvs[0]
        print(f"Single benign feature file — using directly: {feat.name}")
    elif benign_file:
        needle = benign_file.lower().removesuffix(".csv")
        matches = [f for f in feat_csvs if f.stem.lower() == needle]
        if not matches:
            available = [f.name for f in feat_csvs]
            raise FileNotFoundError(
                f"--benign-file '{benign_file}' not found in {feat_dir}. "
                f"Available: {available}"
            )
        feat = matches[0]
        print(f"Using benign feature file: {feat.name}")
    else:
        available = [f.name for f in feat_csvs]
        raise ValueError(
            f"Multiple benign files found in {feat_dir}. "
            f"Specify one with --benign-file. Available: {available}"
        )

    lbl_csvs = sorted(lbl_dir.glob("*.csv")) if lbl_dir.exists() else []
    if lbl_csvs:
        # Match by stem if possible, else fall back to first file
        needle = feat.stem.lower()
        match = next((f for f in lbl_csvs if f.stem.lower() == needle), lbl_csvs[0])
        lbl = match
    else:
        n_rows = sum(1 for _ in open(feat)) - 1
        lbl = tmp_dir / f"{key}_benign_labels.csv"
        _write_temp(pd.DataFrame({"label": [0] * n_rows}), lbl)

    return feat, lbl


def train(args, device):
    data_root = Path(args.data_root)
    artifacts_root = Path(args.artifacts_root)
    mkf = _model_key_full(args)
    dataset_key = f"{args.dataset}_{args.extractor}_{mkf}"

    benign_feat_dir = data_root / args.dataset / "features" / args.extractor / "benign"
    benign_lbl_dir  = data_root / args.dataset / "labels" / "benign"
    tmp_dir = artifacts_root / ".tmp"

    tmp_feat, tmp_lbl = _resolve_benign_csv(
        benign_feat_dir, benign_lbl_dir, args.benign_file, tmp_dir, dataset_key
    )

    StreamingCSVDataset = get_streaming_csv_dataset_cls()
    train_ds = StreamingCSVDataset(
        feature_csv_path=str(tmp_feat),
        label_csv_path=str(tmp_lbl),
        label_column=_LABEL_COLUMN,
        max_samples=args.max_train_samples,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.chdir(_SRC_DIR)
    ids = build_model(args.model, dataset_key, input_size=train_ds.input_size, device=device)
    ids.train_model(train_loader)
    ids.save()

    # Move trained model from titli scratch → checkpoint; delete scratch copy
    scratch_models = _TITLI_SCRATCH / dataset_key / "models"
    ckpt_models = _checkpoint_models_dir(Path(args.checkpoints_root), args.dataset, args.extractor, mkf)
    _stage_models(ckpt_models, scratch_models)
    _cleanup_dir(scratch_models)

    print(f"✓ Model saved to: {ckpt_models}")


# ── Concurrent evaluation helpers ────────────────────────────────────────────

def _collect_eval_tasks(args, data_root: Path) -> list[tuple]:
    """Returns [(split, attack, feat_path, lbl_path), ...] for all eval files."""
    tasks = []
    lbl_malicious_dir = data_root / args.dataset / "labels" / "malicious"
    for split in args.splits:
        if split == "adversarial":
            feat_dir = data_root / args.dataset / "features" / args.extractor / "adversarial" / args.adv_variant
        else:
            feat_dir = data_root / args.dataset / "features" / args.extractor / split
        needle = getattr(args, "malicious_file", None)
        needle = needle.lower().removesuffix(".csv") if needle else None
        for feat_path in sorted(feat_dir.glob("*.csv")):
            if needle and feat_path.stem.lower() != needle:
                continue
            attack = feat_path.stem
            lbl_path = lbl_malicious_dir / f"{attack}.csv"
            if not lbl_path.exists():
                print(f"[skip] missing labels for {attack} (looked in {lbl_path})")
                continue
            tasks.append((split, attack, feat_path, lbl_path))
    return tasks


def _evaluate_file_worker(packed: tuple) -> str:
    """Top-level picklable worker. Returns captured output as a string."""
    (model_key, base_dataset_key, input_size, device_str,
     feat_path_str, lbl_path_str, attack, split,
     results_dir_str, batch_size, max_test_samples,
     src_dir_str, titli_scratch_str, checkpoint_models_dir_str) = packed

    import os, shutil, io, contextlib
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    import score_dump_patch  # noqa: F401 — must run inside the spawned worker too
    from titli_compat import get_streaming_csv_dataset_cls

    src_dir       = Path(src_dir_str)
    titli_scratch = Path(titli_scratch_str)
    results_dir   = Path(results_dir_str)

    # Unique scratch dir per worker so titli's metrics file writes don't collide
    # Stage directly from checkpoint — no dependency on base scratch existing
    worker_key    = f"{base_dataset_key}__{split}__{attack}"
    worker_models = titli_scratch / worker_key / "models"
    _stage_models(worker_models, Path(checkpoint_models_dir_str))

    os.chdir(src_dir)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            StreamingCSVDataset = get_streaming_csv_dataset_cls()
            device = torch.device(device_str)
            ids = build_model(model_key, worker_key, input_size=input_size, device=device)
            ids.load()

            test_ds = StreamingCSVDataset(
                feature_csv_path=feat_path_str,
                label_csv_path=lbl_path_str,
                max_samples=max_test_samples,
                label_column=_LABEL_COLUMN,
            )
            # num_workers=0: no nested multiprocessing inside a subprocess
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            ids.title = attack
            ids.evaluate(test_loader)

            model_dir_name = ids.model_name.lower()
            src_base   = titli_scratch / worker_key
            metrics_src = src_base / "objects" / "metrics" / f"{model_dir_name}.txt"
            scores_src  = src_base / "objects" / "scores"  / f"{model_dir_name}.npz"
            split_dir = results_dir / split
            if metrics_src.exists():
                split_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(metrics_src), str(split_dir / f"{attack}.txt"))
            if scores_src.exists():
                split_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(scores_src), str(split_dir / f"{attack}.npz"))
            for leftover in [
                src_base / "plots" / "confusion_matrix" / f"{model_dir_name}.png",
                src_base / "plots" / "roc"              / f"{model_dir_name}.png",
                src_base / "plots" / "anomaly"          / f"{model_dir_name}.png",
            ]:
                if leftover.exists():
                    leftover.unlink()
    finally:
        worker_scratch = titli_scratch / worker_key
        if worker_scratch.exists():
            shutil.rmtree(str(worker_scratch), ignore_errors=True)

    return f"\n── {split}/{attack} ──\n{buf.getvalue()}"


def evaluate_concurrent(args, device, tasks: list[tuple], input_size: int,
                         dataset_key: str, results_dir: Path) -> None:
    """Dispatch eval tasks to a ProcessPoolExecutor; print each file's output as it completes."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ckpt_models = _checkpoint_models_dir(Path(args.checkpoints_root), args.dataset, args.extractor, _model_key_full(args))
    packed_tasks = [
        (args.model, dataset_key, input_size, str(device),
         str(feat_path), str(lbl_path), attack, split,
         str(results_dir), args.batch_size, args.max_test_samples,
         str(_SRC_DIR), str(_TITLI_SCRATCH), str(ckpt_models))
        for split, attack, feat_path, lbl_path in tasks
    ]

    print(f"Concurrent eval: {len(tasks)} files | workers={args.workers or 'auto'}")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        future_map = {executor.submit(_evaluate_file_worker, p): p for p in packed_tasks}
        for future in as_completed(future_map):
            try:
                print(future.result(), end="", flush=True)
            except Exception as exc:
                p = future_map[future]
                print(f"[ERROR] {p[7]}/{p[6]}: {exc}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args, device):
    data_root = Path(args.data_root)
    artifacts_root = Path(args.artifacts_root)
    mkf = _model_key_full(args)
    dataset_key = f"{args.dataset}_{args.extractor}_{mkf}"
    results_dir = artifacts_root / args.dataset / args.extractor / mkf
    results_dir.mkdir(parents=True, exist_ok=True)

    # Infer input_size from one benign feature file
    benign_feat_dir = data_root / args.dataset / "features" / args.extractor / "benign"
    sample_csv = next(benign_feat_dir.glob("*.csv"), None)
    if sample_csv is None:
        raise FileNotFoundError(f"No benign feature CSVs found in {benign_feat_dir} (needed for input_size)")
    input_size = _get_input_size(sample_csv)

    tasks = _collect_eval_tasks(args, data_root)

    if args.concurrent and len(tasks) > 1:
        evaluate_concurrent(args, device, tasks, input_size, dataset_key, results_dir)
        return

    # ── Sequential path ───────────────────────────────────────────────────────
    ckpt_models = _checkpoint_models_dir(Path(args.checkpoints_root), args.dataset, args.extractor, mkf)
    scratch_models = _TITLI_SCRATCH / dataset_key / "models"
    _stage_models(scratch_models, ckpt_models)
    os.chdir(_SRC_DIR)
    ids = build_model(args.model, dataset_key, input_size=input_size, device=device)
    ids.load()
    _cleanup_dir(scratch_models)

    StreamingCSVDataset = get_streaming_csv_dataset_cls()

    for split, attack, feat_path, lbl_path in tasks:
        print(f"Evaluating {args.model} | {args.extractor} | {split} | {attack}")
        test_ds = StreamingCSVDataset(
            feature_csv_path=str(feat_path),
            label_csv_path=str(lbl_path),
            max_samples=args.max_test_samples,
            label_column=_LABEL_COLUMN,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        ids.title = attack
        ids.evaluate(test_loader)

        model_dir_name = ids.model_name.lower()
        src_base = _TITLI_SCRATCH / dataset_key
        metrics_src = src_base / "objects" / "metrics" / f"{model_dir_name}.txt"
        scores_src  = src_base / "objects" / "scores"  / f"{model_dir_name}.npz"
        split_dir = results_dir / split
        if metrics_src.exists():
            split_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(metrics_src), str(split_dir / f"{attack}.txt"))
        if scores_src.exists():
            split_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(scores_src), str(split_dir / f"{attack}.npz"))

        for leftover in [
            src_base / "plots" / "confusion_matrix" / f"{model_dir_name}.png",
            src_base / "plots" / "roc"              / f"{model_dir_name}.png",
            src_base / "plots" / "anomaly"          / f"{model_dir_name}.png",
        ]:
            if leftover.exists():
                leftover.unlink()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    if args.train:
        train(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
