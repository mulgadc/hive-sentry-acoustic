#!/usr/bin/env python3
"""
Inspect Keras/TensorFlow metadata for our classifier models.

Usage (from repo root):
  python3 scripts/inspect_models.py \
    --models-root /models/models_smoke

Without --models-root, this will try common locations in order:
  1) $MODELS_ROOT
  2) /models/models_smoke  (container default)
  3) src/droneprint/models/models_smoke  (in-repo copy if present)

It will print, for each model file found:
  - path
  - keras_version (from HDF5 attrs)
  - backend
  - top.class_name

Only HDF5 (.h5) models are inspected; .keras or SavedModel folders are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Iterable, List

try:
    import h5py  # type: ignore
except Exception as e:
    print("[inspect_models] h5py is required: pip install h5py", file=sys.stderr)
    raise

# Candidate filenames we expect
EXPECTED_FILES = [
    "classifier_x_binary.h5",
    "classifier_y_make.h5",
]


def pick_models_root(cli_root: str | None) -> Path:
    """Choose a models root directory based on CLI arg, env, or defaults."""
    candidates: List[str] = []
    if cli_root:
        candidates.append(cli_root)
    env_root = os.environ.get("MODELS_ROOT")
    if env_root:
        candidates.append(env_root)
    # container default
    candidates.append("/models/models_smoke")
    # in-repo default
    candidates.append(str(Path(__file__).resolve().parents[1] / "src" / "droneprint" / "models" / "models_smoke"))

    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    # fallback to first
    return Path(candidates[0])


def iter_h5_files(root: Path) -> Iterable[Path]:
    # Prefer expected filenames if present
    for name in EXPECTED_FILES:
        p = root / name
        if p.exists():
            yield p
    # Also include any other .h5 files under root (top-level only)
    try:
        for p in sorted(root.glob("*.h5")):
            if p.name in EXPECTED_FILES:
                continue
            yield p
    except Exception:
        pass


def inspect_h5(path: Path) -> dict:
    info: dict = {"path": str(path), "exists": path.exists(), "keras_version": None, "backend": None, "top_class_name": None}
    if not path.exists():
        return info
    try:
        with h5py.File(path, "r") as f:
            kv = f.attrs.get("keras_version")
            be = f.attrs.get("backend")
            mc = f.attrs.get("model_config")
            info["keras_version"] = (kv.decode("utf-8") if isinstance(kv, (bytes, bytearray)) else kv)
            info["backend"] = (be.decode("utf-8") if isinstance(be, (bytes, bytearray)) else be)
            if mc:
                try:
                    cfg = json.loads(mc.decode("utf-8") if isinstance(mc, (bytes, bytearray)) else mc)
                    info["top_class_name"] = cfg.get("class_name")
                except Exception:
                    pass
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser(description="Inspect Keras .h5 model metadata (keras_version, backend, class).")
    ap.add_argument("--models-root", type=str, default=None, help="Directory containing model files (default: autodetect).")
    args = ap.parse_args()

    root = pick_models_root(args.models_root)
    print(f"[inspect_models] Using models root: {root}")
    if not root.exists():
        print(f"[inspect_models][warn] Models root does not exist: {root}")
        # Show nearby paths to help
        parent = root.parent
        if parent.exists():
            print(f"[inspect_models] Parent directory listing for {parent}:")
            for p in sorted(parent.iterdir()):
                try:
                    print("  -", p)
                except Exception:
                    pass
        sys.exit(1)

    any_found = False
    for p in iter_h5_files(root):
        any_found = True
        info = inspect_h5(p)
        print(f"\n== {info['path']} ==")
        if not info.get("exists"):
            print("MISSING")
            continue
        if "error" in info:
            print("error:", info["error"])
            continue
        print("keras_version:", info.get("keras_version"))
        print("backend:", info.get("backend"))
        print("top.class_name:", info.get("top_class_name"))

    if not any_found:
        print("[inspect_models][warn] No .h5 files found in models root.")
        # Show directory contents to aid debugging
        try:
            print(f"[inspect_models] Directory listing for {root}:")
            for p in sorted(root.iterdir()):
                print("  -", p)
        except Exception:
            pass


if __name__ == "__main__":
    main()
