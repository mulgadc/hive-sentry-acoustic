#!/usr/bin/env python3
"""
Webhook Emitter: decoupled compute-side process that reads audio frames,
computes minimal UI-driving metrics, and POSTs them to a plotting webhook.

It intentionally DOES NOT send GCC curves or TDOA references.

Endpoints expected on the receiver side:
- POST {base_url}/init   -> one-time scene/config and grid info
- POST {base_url}/frame  -> per-frame updates (levels, spectrogram column, SRP map/best, progress, telemetry optional)

Usage examples:
  python webhook_emitter.py \
      --audio-file field_recordings/20250910/250910-T001.WAV \
      --array-ref hex6 \
      --webhook http://127.0.0.1:8000 \
      --fps 15

Notes:
- Geometry is resolved from catalog arrays via --array-ref. Alternatively, you can pass --positions-json.
- SRP map is normalized to [0,1] before sending; set --no-normalize to send raw with min/max.
- This emitter computes GCC internally to run SRP (windowed max) but never sends GCC curves.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from scipy import signal
from scipy.io.wavfile import read as wav_read

# Local modules
from src.config import (
    SAMPLE_RATE,
    FRAME_SIZE,
    NFFT,
    SPEC_NFFT,
    SPEC_DISPLAY_SECONDS,
    SRP_AZ_RANGE,
    SRP_EL_RANGE,
    SRP_GAMMA,
    SRP_UPDATE_INTERVAL,
    SPEED_OF_SOUND_MPS,
)
from src.gcc_phat import compute_gcc_phat_singleblock
from src.srp_phat import compute_srp_phat_windowed_max
from src.catalog_utils import resolve_array_positions_from_catalog
from src.audio_io import load_audio
from src.srp_grid import setup_srp_grid
from src.dsp_utils import (
    make_pair_indices,
    compute_levels,
    compute_spec_power_slice,
    normalize_map,
)

# Avoid non-standard deps; use stdlib urllib for HTTP POST
import urllib.request


def http_post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        # Drain response to avoid connection reuse issues
        _ = resp.read()


# Use shared catalog resolver and DSP utilities


def emit_init_payload(az_grid: np.ndarray, el_grid: np.ndarray,
                      total_frames: int, source_mode: str,
                      gamma: float) -> dict:
    return {
        "version": "1.0",
        "sample_rate_hz": int(SAMPLE_RATE),
        "frame_size": int(FRAME_SIZE),
        "spectrogram": {
            "spec_nfft": int(SPEC_NFFT),
            "display_seconds": float(SPEC_DISPLAY_SECONDS),
        },
        "srp": {
            "az_grid_deg": [float(x) for x in az_grid],
            "el_grid_deg": [float(x) for x in el_grid],
            "gamma_correction": float(gamma),
        },
        "progress": {
            "total_frames": int(total_frames),
            "source_mode": source_mode,
        },
    }

def emit_init_http(base_url: str, payload: dict) -> None:
    http_post_json(base_url.rstrip('/') + '/init', payload)

def emit_init_print(payload: dict) -> None:
    srp = payload.get("srp", {})
    print("[INIT] sample_rate_hz=",
          payload.get("sample_rate_hz"),
          "frame_size=", payload.get("frame_size"),
          "spec_nfft=", payload.get("spectrogram", {}).get("spec_nfft"),
          "srp_grid=az", len(srp.get("az_grid_deg", [])), "x el", len(srp.get("el_grid_deg", [])),
          "total_frames=", payload.get("progress", {}).get("total_frames"))


def build_frame_payload(frame_index: int,
                        rms_first5: List[float],
                        ch0_power_slice: np.ndarray,
                        srp_map: np.ndarray,
                        best_az: float, best_el: float, best_p: float,
                        normalized: bool,
                        pmin: Optional[float], pmax: Optional[float],
                        total_frames: int, source_mode: str,
                        active_gate: bool) -> dict:
    payload = {
        "version": "1.0",
        "frame_index": int(frame_index),
        "audio_time_sec": float(frame_index * FRAME_SIZE / SAMPLE_RATE),
        "levels": {
            "rms_first5": [float(x) for x in rms_first5],
        },
        "spectrogram": {
            "ch0_power_slice": [float(x) for x in ch0_power_slice],
        },
        "srp": {
            "power_map": srp_map.tolist(),
            "best": {
                "azimuth_deg": float(best_az),
                "elevation_deg": float(best_el),
                "power": float(best_p),
            },
            "normalized": bool(normalized),
            "active_gate": bool(active_gate),
        },
        "progress": {
            "total_frames": int(total_frames),
            "source_mode": source_mode,
        },
    }
    if not normalized:
        payload["srp"]["min"] = float(pmin if pmin is not None else 0.0)
        payload["srp"]["max"] = float(pmax if pmax is not None else 1.0)
    return payload

def emit_frame_http(base_url: str, payload: dict) -> None:
    http_post_json(base_url.rstrip('/') + '/frame', payload)

def emit_frame_print(payload: dict) -> None:
    fi = payload.get("frame_index")
    levels = payload.get("levels", {}).get("rms_first5", [])
    spec = payload.get("spectrogram", {}).get("ch0_power_slice", [])
    srp = payload.get("srp", {})
    best = srp.get("best", {})
    pm = srp.get("power_map", [])
    # Compute simple stats without converting big arrays to strings
    spec_len = len(spec)
    spec_max = max(spec) if spec_len > 0 else 0.0
    # srp_map stats
    try:
        import math as _m
        # pm is list of lists; compute min/max safely
        pm_min = min((min(row) for row in pm), default=0.0)
        pm_max = max((max(row) for row in pm), default=0.0)
        el_len = len(pm)
        az_len = len(pm[0]) if el_len > 0 else 0
    except Exception:
        pm_min = pm_max = 0.0
        el_len = az_len = 0
    print(
        f"[FRAME {fi}] RMS={['%.4f'%x for x in levels]} | spec_len={spec_len} vmax={spec_max:.3e} | "
        f"srp_best=({best.get('azimuth_deg')},{best.get('elevation_deg')}) p={best.get('power')} | "
        f"srp_map={az_len}x{el_len} min={pm_min:.3e} max={pm_max:.3e} active={srp.get('active_gate')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Webhook emitter for SRP-PHAT UI payloads (no GCC/TDOA).')
    parser.add_argument('--audio-file', type=str, required=True, help='Path to multi-channel WAV file.')
    parser.add_argument('--array-ref', type=str, required=True, help='Array ID from catalog/arrays/array_defs.json')
    parser.add_argument('--webhook', type=str, required=False, help='Base URL of plotting webhook, e.g., http://127.0.0.1:8000 (omit with --print-only)')
    parser.add_argument('--channels', type=str, default='', help='Comma-separated channel indices to select (default: all from geometry length)')
    parser.add_argument('--fps', type=float, default=15.0, help='Target send/print rate (frames per second)')
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=True, help='Normalize SRP map to [0,1] and apply gamma')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--print-only', action='store_true', help='Do not POST; print concise summaries to stdout')
    args = parser.parse_args()
    if not args.print_only and not args.webhook:
        parser.error("--webhook is required unless --print-only is specified")

    audio_path = args.audio_file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    # Geometry
    positions = resolve_array_positions_from_catalog(args.array_ref)
    num_mics = positions.shape[0]

    # Channel selection
    if args.channels.strip():
        mic_channels = list(map(int, args.channels.split(',')))
    else:
        mic_channels = list(range(num_mics))
    if len(mic_channels) < 5:
        raise ValueError('Please select at least five channels for SRP.')

    # Load audio and extract channels
    audio, _ = load_audio(audio_path, SAMPLE_RATE)  # (samples, channels), sr==SAMPLE_RATE
    if np.max(mic_channels) >= audio.shape[1]:
        raise ValueError(f"Requested channel {np.max(mic_channels)} but file has {audio.shape[1]} channels")
    audio = audio[:, mic_channels]

    # Progress info
    total_frames = int(math.ceil(len(audio) / FRAME_SIZE))
    source_mode = 'WAV'

    # Build SRP grid
    az_min, az_max = tuple(SRP_AZ_RANGE)
    el_min, el_max = tuple(SRP_EL_RANGE)
    az_grid, el_grid, steering_vectors = setup_srp_grid(
        azimuth_range_deg=(float(az_min), float(az_max)),
        elevation_range_deg=(float(el_min), float(el_max)),
        az_step_deg=1.0,
        el_step_deg=1.0,
    )

    # Emit init
    init_payload = emit_init_payload(az_grid, el_grid, total_frames, source_mode, SRP_GAMMA)
    if args.print_only:
        emit_init_print(init_payload)
    else:
        emit_init_http(args.webhook, init_payload)

    # Precompute pair indices (based on selected channels length)
    P = len(mic_channels)
    pair_indices = make_pair_indices(P)

    # Timing control
    target_hop = 1.0 / max(1e-6, args.fps)

    # SRP cadence
    srp_counter = 0

    # Process frames
    for frame_idx in range(total_frames):
        start = frame_idx * FRAME_SIZE
        end = min(start + FRAME_SIZE, len(audio))
        frame = np.zeros((FRAME_SIZE, P), dtype=np.float32)
        frame[:(end - start)] = audio[start:end, :]

        # Compute simple per-channel levels
        rms_first5 = compute_levels(frame, num_series=5)

        # Spectrogram power slice for ch 0
        ch0_power = compute_spec_power_slice(frame[:, 0], SAMPLE_RATE, SPEC_NFFT)

        # Compute GCC curves for all pairs (internal use only)
        # compute_gcc_phat_singleblock expects shape (num_mics, frame_len)
        gcc_curves = compute_gcc_phat_singleblock(frame.T, nfft=NFFT)

        # SRP update
        srp_ran = False
        power_map = None
        best_az = 0.0
        best_el = 0.0
        best_p = 0.0
        srp_counter += 1
        if srp_counter >= int(SRP_UPDATE_INTERVAL):
            srp_counter = 0
            power_map, best_az, best_el, best_p = compute_srp_phat_windowed_max(
                gcc_curves=gcc_curves,
                mic_positions_m=positions,
                pair_indices=pair_indices,
                steering_unit_vectors=steering_vectors,
                azimuth_grid_deg=az_grid,
                elevation_grid_deg=el_grid,
                sampling_rate_hz=SAMPLE_RATE,
                speed_of_sound_mps=float(SPEED_OF_SOUND_MPS),
                nfft=NFFT,
                search_window=5,
            )
            srp_ran = True
        else:
            # If not updated this frame, send a zero map (or keep last? choose zero here for simplicity)
            if power_map is None:
                power_map = np.zeros((len(el_grid), len(az_grid)), dtype=np.float32)

        # Normalize or include min/max
        if args.normalize:
            srp_norm, pmin, pmax = normalize_map(power_map, gamma=float(SRP_GAMMA))
            frame_payload = build_frame_payload(
                frame_idx,
                rms_first5,
                ch0_power,
                srp_norm,
                best_az, best_el, best_p,
                normalized=True,
                pmin=None, pmax=None,
                total_frames=total_frames,
                source_mode=source_mode,
                active_gate=bool(srp_ran),
            )
            if args.print_only:
                emit_frame_print(frame_payload)
            else:
                emit_frame_http(args.webhook, frame_payload)
        else:
            frame_payload = build_frame_payload(
                frame_idx,
                rms_first5,
                ch0_power,
                power_map.astype(np.float32),
                best_az, best_el, best_p,
                normalized=False,
                pmin=float(np.min(power_map)), pmax=float(np.max(power_map)),
                total_frames=total_frames,
                source_mode=source_mode,
                active_gate=bool(srp_ran),
            )
            if args.print_only:
                emit_frame_print(frame_payload)
            else:
                emit_frame_http(args.webhook, frame_payload)

        # crude pacing to approximate FPS
        import time
        time.sleep(max(0.0, target_hop))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
