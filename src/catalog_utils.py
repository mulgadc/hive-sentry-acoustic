from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def project_root() -> Path:
    """Return the project root directory (where 'src/' and 'catalog/' live)."""
    # This module resides in src/, so parent is project root
    return Path(__file__).resolve().parent.parent


def resolve_array_positions_from_catalog(array_ref: str) -> np.ndarray:
    """Load microphone positions for a given array ID from catalog/arrays/array_defs.json.

    Returns a numpy array of shape (num_mics, 3) with float dtype.
    Raises FileNotFoundError or ValueError on errors.
    """
    repo_root = project_root()
    arrays_path = repo_root / 'catalog' / 'arrays' / 'array_defs.json'
    if not arrays_path.exists():
        raise FileNotFoundError(f"Arrays catalog not found: {arrays_path}")
    with open(arrays_path, 'r') as f:
        arrays_doc = json.load(f)
    for a in arrays_doc.get('arrays', []):
        if a.get('id') == array_ref:
            positions = np.array(a.get('positions', []), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
                raise ValueError(f"Array '{array_ref}' has invalid positions shape: {positions.shape}")
            return positions
    raise ValueError(f"Array ref '{array_ref}' not found in {arrays_path}")
