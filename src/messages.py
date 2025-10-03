from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np


@dataclass
class AudioFrame:
    idx: int
    ts: float
    audio: np.ndarray  # shape (FRAME_SIZE, channels), float32
    sr: int


@dataclass
class Detection:
    active: bool
    scores: Dict[str, float]
    thresholds: Dict[str, float]


@dataclass
class DOAResult:
    azimuth_deg: float
    elevation_deg: float
    power: float
    power_map: Optional[np.ndarray]
    az_grid_deg: Optional[np.ndarray]
    el_grid_deg: Optional[np.ndarray]
    active_gate: bool


@dataclass
class Classification:
    label: str
    prob: float
    logits: Optional[List[float]] = None


@dataclass
class FrameResult:
    frame_idx: int
    detection: Detection
    doa: DOAResult
    classification: Optional[Classification]
    meta: Dict
    levels_first5: Optional[List[float]] = None
    spec_power_mic0: Optional[List[float]] = None
