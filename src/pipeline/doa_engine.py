from __future__ import annotations

import numpy as np
from typing import Tuple

from src.config import SAMPLE_RATE, NFFT, SRP_UPDATE_INTERVAL, SRP_GAMMA, SPEED_OF_SOUND_MPS
import os, time
from src.srp_grid import setup_srp_phat_grid_context
from src.gcc_phat import compute_gcc_phat_singleblock
from src.srp_phat import compute_srp_phat_windowed_max
from src.dsp_utils import make_pair_indices
from src.messages import AudioFrame, DOAResult
from src.nd_backend import xp, asarray  # backend arrays (NumPy/CuPy)


class DOAEngine:
    """SRP-PHAT Direction-of-Arrival engine.

    Prepares a steering grid and, on process(), computes GCC-PHAT and SRP-PHAT windowed maxima.
    """
    def __init__(
        self,
        array_positions: np.ndarray,
        az_range: Tuple[float, float],
        el_range: Tuple[float, float],
        az_step: float = 1.0,
        el_step: float = 1.0,
        update_interval: int = SRP_UPDATE_INTERVAL,
        gamma: float = SRP_GAMMA,
    ):
        self.array_positions = np.asarray(array_positions, dtype=float)
        self.az_range = tuple(az_range)
        self.el_range = tuple(el_range)
        self.az_step = float(az_step)
        self.el_step = float(el_step)
        self.update_interval = int(update_interval)
        self.gamma = float(gamma)

        self.az_grid = None
        self.el_grid = None
        self.steering_vectors = None
        self._counter = 0
        # Precomputed geometry-dependent indices to avoid per-frame recompute
        self._pre_idx_all = None  # shape (P, Q) int32 indices into GCC curves
        self._pre_offsets = None  # shape (W,) window offsets
        self._pair_m1_b = None    # backend array for pair first indices (P,)
        self._pair_m2_b = None    # backend array for pair second indices (P,)
        self._idx_pw_b = None     # backend precomputed (P,Q,W) indices for gather

    def prepare(self):
        # Build SRP-PHAT grid and steering vectors exactly like the viewer
        (
            self.az_grid,
            self.el_grid,
            self.steering_vectors,
            _srp_power_map_init,
            _summary,
        ) = setup_srp_phat_grid_context(
            azimuth_range_deg=self.az_range,
            elevation_range_deg=self.el_range,
            az_step_deg=self.az_step,
            el_step_deg=self.el_step,
        )
        # Precompute microphone pair indices once (viewer uses all i<j pairs)
        n_mics = int(self.array_positions.shape[0])
        self._pair_indices = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]
        # Backend copies for fast indexing in GCC
        m1_idx = np.array([i for (i, _) in self._pair_indices], dtype=np.int32)
        m2_idx = np.array([j for (_, j) in self._pair_indices], dtype=np.int32)
        self._pair_m1_b = asarray(m1_idx)
        self._pair_m2_b = asarray(m2_idx)

        # Precompute expected-delay indices (pair, direction) and window offsets once
        # Shapes: P pairs, Q directions
        P = len(self._pair_indices)
        Q = int(self.steering_vectors.shape[0]) if self.steering_vectors is not None else 0
        if P > 0 and Q > 0:
            center = int(NFFT) // 2
            # Baselines for all pairs: (P, 3)
            baselines = np.array([
                self.array_positions[j] - self.array_positions[i]
                for (i, j) in self._pair_indices
            ], dtype=float)
            # Project baselines onto steering directions -> distances (P, Q)
            proj = baselines @ self.steering_vectors.T  # (P, Q)
            tau_samples_all = (proj / float(SPEED_OF_SOUND_MPS)) * float(SAMPLE_RATE)
            self._pre_idx_all = np.rint(tau_samples_all).astype(np.int32) + center  # (P, Q)
            # Window offsets for search_window=5 (matches current call)
            W = int(5)
            self._pre_offsets = np.arange(-W, W + 1, dtype=np.int32)
            # Precompute (P,Q,W) indices on backend to avoid per-frame build
            idx_all_b = asarray(self._pre_idx_all)
            offsets_b = asarray(self._pre_offsets)
            idx_pw = idx_all_b[:, :, None] + offsets_b[None, None, :]
            # Clip in-place on backend
            xp.clip(idx_pw, 0, int(NFFT) - 1, out=idx_pw)
            self._idx_pw_b = idx_pw

    def process(self, frame: AudioFrame, active_gate: bool) -> DOAResult:
        sig = frame.audio  # (samples, channels)
        # GCC-PHAT expects (num_mics, frame_len)
        _perf_on = os.environ.get('PERF_PROFILE', '0') in ('1', 'true', 'True')
        try:
            _perf_every = max(1, int(os.environ.get('PERF_EVERY', '30')))
        except Exception:
            _perf_every = 30
        t0 = time.perf_counter() if _perf_on else 0.0
        gcc_curves = compute_gcc_phat_singleblock(sig.T, nfft=NFFT, pair_m1_b=self._pair_m1_b, pair_m2_b=self._pair_m2_b)
        t1 = time.perf_counter() if _perf_on else 0.0

        ran = False
        power_map = None
        best_az = 0.0
        best_el = 0.0
        best_power = 0.0

        self._counter += 1
        if active_gate and (self._counter >= self.update_interval):
            self._counter = 0
            # Use precomputed full set of pairs to match viewer behavior
            pairs = getattr(self, "_pair_indices", None)
            if not pairs:
                pairs = make_pair_indices(sig.shape[1])
            power_map, best_az, best_el, best_power = compute_srp_phat_windowed_max(
                gcc_curves=gcc_curves,
                mic_positions_m=self.array_positions,
                pair_indices=pairs,
                steering_unit_vectors=self.steering_vectors,
                azimuth_grid_deg=self.az_grid,
                elevation_grid_deg=self.el_grid,
                sampling_rate_hz=SAMPLE_RATE,
                speed_of_sound_mps=float(SPEED_OF_SOUND_MPS),
                nfft=NFFT,
                search_window=5,
                pre_idx_all=self._pre_idx_all,
                pre_offsets=self._pre_offsets,
                pre_idx_pw=self._idx_pw_b,
            )
            ran = True
            if _perf_on and (int(frame.index) % _perf_every == 0):
                t2 = time.perf_counter()
                gcc_ms = (t1 - t0) * 1000.0
                srp_ms = (t2 - t1) * 1000.0
                print(f"[DOA_PERF] frame={int(frame.index)} gcc={gcc_ms:.2f}ms srp={srp_ms:.2f}ms total={(gcc_ms+srp_ms):.2f}ms Q={len(self.az_grid)*len(self.el_grid)} P={len(pairs)} nfft={NFFT}")

        return DOAResult(
            azimuth_deg=float(best_az),
            elevation_deg=float(best_el),
            power=float(best_power),
            power_map=power_map.astype(np.float32) if power_map is not None else None,
            az_grid_deg=self.az_grid,
            el_grid_deg=self.el_grid,
            active_gate=ran,
        )
