#!/usr/bin/env python3
"""
Multi-microphone GCC-PHAT viewer with SRP-PHAT direction-of-arrival estimation.
Analyzes pre-recorded audio files with 5+ channel support and SRP-PHAT beamforming.
VERSION: 4.0 - SRP-PHAT INTEGRATION
"""

VERSION = "4.0"
print(f"=== SRP-PHAT GCC Viewer v{VERSION} ===")

import numpy as np
import matplotlib as mpl
# Choose a safe backend before importing pyplot to avoid Qt/xcb issues
try:
    import os as _os
    if not _os.environ.get('DISPLAY'):
        # Headless or no X server
        mpl.use('Agg', force=True)
    else:
        # Prefer TkAgg to avoid Qt dependency; fallback to Agg
        try:
            mpl.use('TkAgg', force=True)
        except Exception:
            mpl.use('Agg', force=True)
except Exception:
    # Last resort: let matplotlib decide
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
from scipy import signal
from scipy.io.wavfile import read
import librosa
import os
import threading
import time
import queue
from datetime import datetime, timedelta, timezone
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

# Add src directory to path for SRP-PHAT imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Centralized configuration constants
from src.config import (
    SAMPLE_RATE,
    FRAME_SIZE,
    NFFT,
    UPDATE_INTERVAL,
    SPEC_DISPLAY_SECONDS,
    SPEC_NFFT,
    LOWPASS_FREQ,
    HIGHPASS_FREQ,
    AUDIO_CHUNK_SIZE,
    SPEED_OF_SOUND_MPS,
    SRP_ENABLED_DEFAULT,
    SRP_UPDATE_INTERVAL,
    SRP_GAMMA,
    SRP_AVG_FRAMES,
    SRP_ACTIVITY_RMS_THRESH,
    SRP_GATE_HOLD_SEC,
    SRP_AZ_STEP_DEG_DEFAULT,
    SRP_EL_STEP_DEG_DEFAULT,
    SRP_AZ_RANGE,
    SRP_EL_RANGE,
    VIEWER_FPS_DEFAULT,
    SKIP_INTERMEDIATE_DEFAULT,
    DEFAULT_ARRAY_LAT,
    DEFAULT_ARRAY_LON,
    DEFAULT_ARRAY_ALT,
    MODELS_ROOT as CONFIG_MODELS_ROOT,
)

# Helper utilities (geometry/tdoa without side-effects)
from src.helpers import azimuth_elevation_to_unit_vector, expected_pair_lag_samples
from src.audio_io import load_audio
from src.catalog_utils import resolve_array_positions_from_catalog
from src.srp_grid import setup_srp_grid, setup_srp_phat_grid_context
from src.dsp_utils import compute_levels as dsp_compute_levels, compute_spec_power_slice as dsp_compute_spec_power_slice
from src.nd_backend import xp, asnumpy  # backend-aware ops (NumPy/CuPy)

try:
    from src.srp_phat import compute_srp_phat_for_frame
    from src.srp_phat import compute_srp_phat_windowed_max
    from src.build_steering_grid import build_steering_grid
    from src.gcc_phat import compute_gcc_phat_singleblock
    SRP_PHAT_AVAILABLE = True
    print("✓ SRP-PHAT modules imported successfully")
except ImportError as e:
    SRP_PHAT_AVAILABLE = False
    print(f"✗ SRP-PHAT modules not available: {e}")
    print("  SRP-PHAT functionality disabled")

# Optional: TensorFlow/Keras for classifier .h5 models
try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
    print("✓ TensorFlow available - classifier can be enabled if models exist")
except Exception:
    TF_AVAILABLE = False
    print("✗ TensorFlow not available - classifier disabled (install tensorflow to enable)")

# Telemetry and WAV BWF parsing (optional, non-realtime overlay)
try:
    from src.telemetry import Telemetry, geodetic_to_ecef, ecef_to_enu
    from src.wav_bwf import get_wav_start_utc
    TELEMETRY_AVAILABLE = True
    print("✓ Telemetry modules imported successfully")
except Exception as e:
    TELEMETRY_AVAILABLE = False
    print(f"✗ Telemetry modules not available: {e}")

# Try to import sounddevice
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
    print("✓ sounddevice imported successfully - Audio playback available")
except ImportError:
    AUDIO_AVAILABLE = False
    print("✗ sounddevice not available - Audio playback disabled")
    print("  Install with: pip install sounddevice")

# --- Source modes ---
SOURCE_WAV = 'WAV'
SOURCE_STREAM = 'STREAM'

# No fallback array geometry: array positions must come from the catalog

def rotate_array(positions, azimuth_degrees):
    """Rotate array positions by azimuth angle"""
    angle_rad = np.radians(-azimuth_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])
    return positions @ rotation_matrix.T


# Configuration values are imported from src.config

class SRPGCCViewer:
    def __init__(self, audio_file_path, mic_channels=None, playback_speed=1.0, enable_audio=True, playback_device=None, array_heading_deg: float = 0.0,
                 source_mode: str = SOURCE_WAV, input_device=None, list_devices: bool = False,
                 telemetry_path: str | None = None, array_lat: float | None = None, array_lon: float | None = None, array_alt: float | None = None,
                 audio_local_utc_offset: str = "+10:00", telemetry_auto: bool = True,
                 display_fps: float = 15.0, skip_intermediate: bool = True,
                 az_step_deg: float = 1.0, el_step_deg: float = 1.0,
                 array_ref: str | None = None):
        print(f"Initializing Multi-Mic GCC Viewer v{VERSION}...")
        self.audio_file_path = audio_file_path
        # Geometry must come from catalog (no fallback). Initialize empty and set once resolved.
        self.array_positions = np.zeros((0, 3), dtype=float)
        self._geometry_set = False
        self.array_ref = array_ref
        # Track whether CLI provided channels so we can default after geometry
        _cli_channels_provided = (mic_channels is not None)
        # If channels not provided, temporarily use first 5 indices; will reset after geometry if needed
        self.mic_channels = mic_channels if _cli_channels_provided else [0, 1, 2, 3, 4]
        # Ensure geometry has at least as many rows as selected channels
        self._ensure_geometry_capacity()
        self.playback_speed = playback_speed
        self.enable_audio = False  # Default to audio OFF
        self.playback_device = playback_device
        # Track whether CLI provided array fields (so we don't override from catalog)
        _cli_array_coords_provided = (array_lat is not None) or (array_lon is not None) or (array_alt is not None)
        self.array_heading_deg = float(array_heading_deg)
        self.source_mode = source_mode if source_mode in (SOURCE_WAV, SOURCE_STREAM) else SOURCE_WAV
        self.input_device = input_device
        self.list_devices = list_devices
        # Telemetry related
        self.telemetry_path = telemetry_path
        self.array_lat = float(array_lat) if array_lat is not None else float(DEFAULT_ARRAY_LAT)
        self.array_lon = float(array_lon) if array_lon is not None else float(DEFAULT_ARRAY_LON)
        self.array_alt = float(array_alt) if array_alt is not None else float(DEFAULT_ARRAY_ALT)
        self.audio_local_utc_offset = audio_local_utc_offset
        self.telemetry = None
        self.audio_start_utc = None
        # Telemetry visualization handles (target-like symbol)
        self.tele_marker = None  # legacy handle, unused after target upgrade
        self.tele_marker_outer = None
        self.tele_marker_inner = None
        self.telemetry_auto = telemetry_auto
        
        # TDOA reference parameters
        self.target_azimuth = 90.0    # Default to 90° azimuth (East)
        self.target_elevation = 0.0   # Default to 0° elevation (horizontal)
        # Ensure geometry reflects heading (recompute after possible catalog override later)
        self.show_tdoa_reference = True
        
        # SRP-PHAT parameters (defaults from config, gated by availability)
        self.enable_srp_phat = SRP_PHAT_AVAILABLE and bool(SRP_ENABLED_DEFAULT)
        self.srp_update_interval = int(SRP_UPDATE_INTERVAL)  # e.g., update every N frames
        self.srp_frame_counter = 0
        self.gamma_correction = float(SRP_GAMMA)  # Gamma for power-law scaling (lower = more peak emphasis)
        self.srp_averaging_frames = int(SRP_AVG_FRAMES)  # Averaging for smoothing
        self.srp_power_history = deque(maxlen=self.srp_averaging_frames)
        # PERF: track SRP timing across frames to provide amortized stats
        self._last_srp_time_ms = 0.0
        self._last_srp_ran = False
        # Activity gating to prevent random static when quiet
        hop_sec = FRAME_SIZE / SAMPLE_RATE
        self.srp_activity_rms_thresh = float(SRP_ACTIVITY_RMS_THRESH)
        self.srp_gate_hold_frames = max(1, int(float(SRP_GATE_HOLD_SEC) / hop_sec))
        self._srp_gate_counter = 0
        self._srp_last_active = False
        # PERF: track real-time pacing based on wall-clock between progress updates
        self._rt_last_time = None
        self._rt_last_frame = 0
        self._rt_log_count = 0
        self._rt_ema = None  # exponential moving average of RTF
        self._rt_ema_alpha = 0.8
        # Real-time processing pacing
        self._rt_start_time = None  # wall-clock when playback starts/resumes
        # Display frame pacing
        self.display_fps = float(display_fps) if display_fps and display_fps > 0 else float(VIEWER_FPS_DEFAULT)
        self.skip_intermediate = bool(skip_intermediate)
        # SRP grid resolution (degrees)
        self.az_step_deg = max(0.1, float(az_step_deg))
        self.el_step_deg = max(0.1, float(el_step_deg))
        # DOA timing telemetry (instrumentation only)
        self._doa_log_last_ts = None
        self._doa_log_frames = 0
        self._gcc_ms_accum = 0.0
        self._srp_ms_accum = 0.0
        self._doa_ms_accum = 0.0
        
        # Additional filtering options
        self.use_bandpass_filter = True  # Toggle for 300-3400 Hz bandpass
        self.use_lowpass_filter = False  # Toggle for additional low-pass filtering
        if self.enable_srp_phat:
            self.setup_srp_phat_grid()
        else:
            self.srp_power_map = None
        
        if enable_audio and not AUDIO_AVAILABLE:
            print("Audio requested but sounddevice not available")
        elif self.enable_audio:
            try:
                # Test audio system
                _ = sd.query_devices()
                print("✓ Audio system ready")
            except Exception as e:
                print(f"✗ Audio system test failed: {e}")
                self.enable_audio = False
        
        # STREAM-specific: setup input queue and device
        self.input_queue = None
        self.input_stream = None
        self.stream_stop_event = threading.Event()
        self.stream_blocksize = FRAME_SIZE
        self.stream_device_name = None

        if self.source_mode == SOURCE_STREAM:
            if not AUDIO_AVAILABLE:
                raise RuntimeError("STREAM mode requires sounddevice; please install it.")
            # STREAM mode: enforce explicit array selection from catalog
            if not self.array_ref:
                raise RuntimeError("STREAM mode requires an explicit array selection. Provide --array-ref <ARRAY_ID> from catalog/arrays/array_defs.json.")
            try:
                positions = resolve_array_positions_from_catalog(self.array_ref)
                if positions is None or positions.size == 0:
                    raise RuntimeError(f"Array ref '{self.array_ref}' not found or has no positions in catalog/arrays/array_defs.json")
                # Apply heading rotation if provided
                self.array_positions = rotate_array(positions, self.array_heading_deg) if self.array_heading_deg != 0.0 else positions
                self._geometry_set = True
                # Default channels to geometry length if not explicitly provided
                if not _cli_channels_provided:
                    self.mic_channels = list(range(self.array_positions.shape[0]))
            except Exception as e:
                raise RuntimeError(f"Failed to resolve array geometry for STREAM mode: {e}")
            if self.list_devices:
                self._print_audio_devices()
            self._select_input_device()
            self._setup_input_stream_infrastructure()
            # In STREAM mode, we do not preload any file
            self.total_frames = 0
        else:
            self.load_audio_file()
            self.total_frames = len(self.audio_data) // FRAME_SIZE
            # Auto-select telemetry from catalog BEFORE loading, so it gets picked up
            if (self.telemetry_path is None) and getattr(self, 'telemetry_auto', True):
                try:
                    auto_path = self._resolve_telemetry_from_catalog(self.audio_file_path)
                    if auto_path:
                        self.telemetry_path = auto_path
                        print(f"✓ Telemetry auto-selected from catalog: {self.telemetry_path}")
                except Exception as e:
                    print(f"⚠️ Telemetry auto-select failed: {e}")
            # Auto-populate array coordinates and heading from the catalog recording, unless CLI provided
            try:
                repo_root = Path(__file__).resolve().parent
                cat_rec_dir = repo_root / 'catalog' / 'recordings'
                rec_doc = None
                if cat_rec_dir.exists():
                    # Try to match by absolute or repo-relative path
                    wav_abs = str(Path(self.audio_file_path).resolve())
                    wav_rel = None
                    try:
                        wav_rel = str(Path(self.audio_file_path).resolve().relative_to(repo_root))
                    except Exception:
                        pass
                    for p in cat_rec_dir.glob('*.json'):
                        try:
                            with open(p, 'r') as f:
                                doc = json.load(f)
                            path_field = doc.get('path', '')
                            # Build candidate matches
                            candidates = [path_field]
                            try:
                                candidates.append(str((repo_root / path_field).resolve()))
                            except Exception:
                                pass
                            if wav_abs in candidates or (wav_rel is not None and wav_rel == path_field):
                                rec_doc = doc
                                break
                        except Exception:
                            continue
                if rec_doc and isinstance(rec_doc.get('array'), dict):
                    arr = rec_doc['array']
                    # Coordinates
                    if not _cli_array_coords_provided:
                        if 'lat' in arr:
                            self.array_lat = float(arr['lat'])
                        if 'lon' in arr:
                            self.array_lon = float(arr['lon'])
                        if 'alt_m' in arr:
                            self.array_alt = float(arr['alt_m'])
                        print(f"✓ Array coordinates from catalog: lat={self.array_lat}, lon={self.array_lon}, alt_m={self.array_alt}")
                    # Heading (only override if default 0.0 or CLI likely not set)
                    if 'heading_deg' in arr and (self.array_heading_deg == 0.0):
                        self.array_heading_deg = float(arr['heading_deg'])
                        print(f"✓ Array heading from catalog: {self.array_heading_deg}°")

                    # Geometry from catalog: prefer explicit positions, else resolve ref from arrays registry
                    try:
                        positions = None
                        if isinstance(arr.get('positions'), list) and len(arr['positions']) > 0:
                            positions = np.array(arr['positions'], dtype=float)
                            print(f"✓ Using {positions.shape[0]} mic positions from recording doc")
                        elif isinstance(arr.get('ref'), str):
                            arrays_path = repo_root / 'catalog' / 'arrays' / 'array_defs.json'
                            with open(arrays_path, 'r') as f:
                                arrays_doc = json.load(f)
                            arr_id = arr['ref']
                            found = None
                            for a in arrays_doc.get('arrays', []):
                                if a.get('id') == arr_id:
                                    found = a
                                    break
                            if found is None:
                                print(f"⚠️ Array ref '{arr_id}' not found in {arrays_path}")
                            else:
                                positions = np.array(found.get('positions', []), dtype=float)
                                print(f"✓ Resolved array.ref='{arr_id}' to {positions.shape[0]} positions")
                        # If positions were provided or resolved, apply heading and set array_positions
                        if positions is not None and positions.size > 0:
                            self.array_positions = rotate_array(positions, self.array_heading_deg) if self.array_heading_deg != 0.0 else positions
                            self._geometry_set = True
                            # If CLI did not provide channels, use channel_map or default to all
                            if not _cli_channels_provided:
                                if isinstance(arr.get('channel_map'), list) and len(arr['channel_map']) > 0:
                                    self.mic_channels = list(map(int, arr['channel_map']))
                                    print(f"✓ Using channel_map from recording: {self.mic_channels}")
                                else:
                                    self.mic_channels = list(range(self.array_positions.shape[0]))
                                    print(f"✓ Defaulting mic channels to 0..{self.array_positions.shape[0]-1}")
                            # Validate sizes
                            if self.array_positions.shape[0] < len(self.mic_channels):
                                print(f"⚠️ Catalog geometry ({self.array_positions.shape[0]}) has fewer mics than selected channels ({len(self.mic_channels)}); padding geometry.")
                                self._ensure_geometry_capacity()
                    except Exception as e:
                        print(f"⚠️ Could not load geometry from catalog: {e}")
            except Exception as e:
                print(f"⚠️ Could not resolve array settings from catalog: {e}")
            # Require geometry for WAV mode; fail fast if not set
            if not self._geometry_set:
                raise RuntimeError("No array geometry found for WAV mode. Ensure the recording's catalog JSON embeds positions or references an array in catalog/arrays/array_defs.json.")
            # If CLI channels were not provided and not set from catalog, default to all from geometry
            if (not _cli_channels_provided) and (self.mic_channels is None or len(self.mic_channels) == 0):
                self.mic_channels = list(range(self.array_positions.shape[0]))
            # Ensure geometry can accommodate selected channels
            self._ensure_geometry_capacity()
            # Load telemetry in WAV mode only
            if TELEMETRY_AVAILABLE and self.telemetry_path:
                try:
                    print(f"Loading telemetry from {self.telemetry_path}")
                    self.telemetry = Telemetry.load_csv(self.telemetry_path)
                except Exception as e:
                    print(f"✗ Failed to load telemetry: {e}")
                    self.telemetry = None
            # Determine audio start UTC from BWF metadata using local offset
            if TELEMETRY_AVAILABLE and self.source_mode == SOURCE_WAV:
                try:
                    self.audio_start_utc = get_wav_start_utc(self.audio_file_path, self.audio_local_utc_offset)
                    if self.audio_start_utc is None:
                        print("⚠️ Could not determine audio start UTC from BWF; telemetry overlay will be disabled.")
                    else:
                        # Print start time in UTC and local (per provided offset)
                        print(f"✓ Audio start time (UTC): {self.audio_start_utc.isoformat()}")
                        try:
                            s = self.audio_local_utc_offset.strip()
                            sign = 1
                            if s.startswith('-'):
                                sign = -1
                                s = s[1:]
                            elif s.startswith('+'):
                                s = s[1:]
                            if ':' in s:
                                h, m = s.split(':', 1)
                                oh, om = int(h), int(m)
                            else:
                                oh, om = int(s), 0
                            tz = timezone(sign * timedelta(hours=oh, minutes=om))
                            print(f"✓ Audio start time (local {self.audio_local_utc_offset}): {self.audio_start_utc.astimezone(tz).isoformat()}")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️ Audio start UTC computation failed: {e}")
                    self.audio_start_utc = None

            # Compute and log telemetry/audio time ranges and overlap
            self.telemetry_overlap = False
            try:
                if (self.telemetry is not None) and (self.audio_start_utc is not None):
                    from datetime import timedelta as _td
                    tele_start = self.telemetry.times[0]
                    tele_end = self.telemetry.times[-1]
                    audio_dur_sec = (len(self.audio_data) / float(SAMPLE_RATE)) if hasattr(self, 'audio_data') else 0.0
                    audio_start = self.audio_start_utc
                    audio_end = audio_start + _td(seconds=audio_dur_sec)
                    print(f"Telemetry UTC range: {tele_start.isoformat()} -> {tele_end.isoformat()}")
                    print(f"Audio UTC range:     {audio_start.isoformat()} -> {audio_end.isoformat()} (dur {audio_dur_sec:.2f}s)")
                    latest_start = max(tele_start, audio_start)
                    earliest_end = min(tele_end, audio_end)
                    self.telemetry_overlap = (earliest_end >= latest_start)
                    print(f"Overlap: {'YES' if self.telemetry_overlap else 'NO'}")
            except Exception as e:
                print(f"⚠️ Could not compute telemetry/audio overlap: {e}")

        # Attempt to resolve telemetry automatically via catalog if not provided
        if (self.telemetry_path is None) and self.telemetry_auto:
            try:
                self.telemetry_path = self._resolve_telemetry_from_catalog(self.audio_file_path)
                if self.telemetry_path:
                    print(f"✓ Telemetry auto-selected from catalog: {self.telemetry_path}")
            except Exception as e:
                print(f"⚠️ Telemetry auto-select failed: {e}")

        self.current_frame_idx = 0
        self.is_playing = False
        self.is_paused = False
        
        # Audio playback - simpler threaded approach
        self.audio_stream = None
        self.audio_thread = None
        self.audio_stop_event = threading.Event()
        self.audio_play_position = 0  # Track audio position separately
        # Lock to coordinate audio position updates between UI and audio callback
        self._audio_lock = threading.Lock()
        # Real-time pacing anchor (ensures consistent speed after seeks)
        self._rt_anchor_time = None  # wall-clock time when anchor set
        self._rt_anchor_frame = 0    # visual frame index at anchor time
        
        self.setup_buffers()
        self.setup_filters()
        self.setup_plots()

        # Optional classifier integration (runs in background)
        try:
            self._init_classifier()
        except Exception as e:
            print(f"⚠️ Classifier init failed: {e}")

    def load_audio_file(self):
        """Load audio file and extract the specified channels"""
        print(f"Loading audio file: {self.audio_file_path}")
        # Use shared loader (returns float32, shape (samples, channels), at SAMPLE_RATE)
        try:
            audio_data, sr = load_audio(self.audio_file_path, SAMPLE_RATE)
        except Exception as e:
            raise ValueError(f"Could not load audio file: {self.audio_file_path} ({e})")
        print(f"Loaded audio: {sr} Hz, {audio_data.shape} shape")
        
        # Extract specified channels
        if audio_data.ndim == 1:
            if len(self.mic_channels) > 1:
                print("Warning: Mono file loaded but multiple channels requested. Duplicating channel.")
                # Create 5 channels from mono for testing
                self.audio_data = np.column_stack([audio_data] * 5)
            else:
                self.audio_data = audio_data.reshape(-1, 1)
        else:
            if max(self.mic_channels) >= audio_data.shape[1]:
                raise ValueError(f"Requested channel {max(self.mic_channels)} but file only has {audio_data.shape[1]} channels")
            self.audio_data = audio_data[:, self.mic_channels]
        
        print(f"Using channels {self.mic_channels}, final shape: {self.audio_data.shape}")
        
        # Ensure we have at least 5 channels for multi-mic GCC-PHAT
        if self.audio_data.shape[1] < 5:
            raise ValueError("Multi-mic GCC-PHAT requires at least 5 channels")

    def setup_buffers(self):
        # Build microphone pairs dynamically from selected channels
        ch = list(range(len(self.mic_channels)))
        self.pair_indices = [(i, j) for i in ch for j in ch if i < j]
        self.pair_labels = [f"{i}-{j}" for (i, j) in self.pair_indices]
        self.gcc_data_pairs = [np.zeros(NFFT) for _ in range(len(self.pair_indices))]
        self.lag_axis = np.arange(-NFFT//2, NFFT//2)
        self.level_history = deque(maxlen=100)
        self.spec_frames_to_keep = int(SPEC_DISPLAY_SECONDS * SAMPLE_RATE / FRAME_SIZE)
        # Only keep spectrogram for channel 0
        self.spec_history_mic0 = deque(maxlen=self.spec_frames_to_keep)
        self.spec_freqs = np.fft.rfftfreq(SPEC_NFFT, 1/SAMPLE_RATE)

    def setup_filters(self):
        nyquist = SAMPLE_RATE / 2
        
        # Original bandpass filter (100-8000 Hz)
        self.sos_original = signal.butter(4, [HIGHPASS_FREQ / nyquist, LOWPASS_FREQ / nyquist], btype='band', output='sos')
        
        # Voice bandpass filter (300-3400 Hz)
        self.sos_bandpass = signal.butter(4, [300 / nyquist, 3400 / nyquist], btype='band', output='sos')
        
        # Low-pass filter (for spatial anti-aliasing - prevents aliasing when wavelength ≈ mic spacing)
        self.sos_lowpass = signal.butter(4, 300 / nyquist, btype='low', output='sos')
        
        # Initialize streaming filter state (per channel, per section)
        num_ch = len(self.mic_channels)
        self._zi_original = np.zeros((num_ch, self.sos_original.shape[0], 2), dtype=float)
        self._zi_bandpass = np.zeros((num_ch, self.sos_bandpass.shape[0], 2), dtype=float)
        self._zi_lowpass = np.zeros((num_ch, self.sos_lowpass.shape[0], 2), dtype=float)

    def apply_filter(self, audio_data):
        """Streaming-safe filtering with persistent zi state per channel.
        Expects audio_data shape (samples, channels).
        """
        # Ensure 2D
        if audio_data.ndim == 1:
            audio_data = audio_data[:, None]
        samples, channels = audio_data.shape
        out = audio_data.astype(np.float32, copy=True)
        # Original band (always on)
        for ch in range(min(channels, len(self.mic_channels))):
            out[:, ch], self._zi_original[ch] = signal.sosfilt(self.sos_original, out[:, ch], zi=self._zi_original[ch])
        # Optional bandpass voice
        if self.use_bandpass_filter:
            for ch in range(min(channels, len(self.mic_channels))):
                out[:, ch], self._zi_bandpass[ch] = signal.sosfilt(self.sos_bandpass, out[:, ch], zi=self._zi_bandpass[ch])
        # Optional lowpass
        if self.use_lowpass_filter:
            for ch in range(min(channels, len(self.mic_channels))):
                out[:, ch], self._zi_lowpass[ch] = signal.sosfilt(self.sos_lowpass, out[:, ch], zi=self._zi_lowpass[ch])
        return out

    def _ensure_geometry_capacity(self):
        """Ensure `self.array_positions` has at least as many microphones as `self.mic_channels`.
        If not, append reasonable defaults:
        - If exactly one extra mic is needed and we have 5 defined, append a 'second spike' at z≈2.0 m.
        - Otherwise append zeros for remaining to satisfy indexing.
        """
        try:
            needed = len(self.mic_channels)
            have = int(self.array_positions.shape[0]) if hasattr(self, 'array_positions') else 0
            if have < needed:
                add = needed - have
                extras = np.zeros((add, 3), dtype=float)
                if have >= 5 and add >= 1:
                    # Add a second spike above/below, keep sign of existing spike's z if available
                    z_sign = 1.0
                    try:
                        z_sign = 1.0 if self.array_positions[4, 2] >= 0 else -1.0
                    except Exception:
                        pass
                    extras[0] = np.array([0.0, 0.0, 2.0 * z_sign], dtype=float)
                self.array_positions = np.vstack([self.array_positions, extras])
        except Exception:
            # Fail-safe: do nothing if geometry cannot be adjusted
            pass
    
    def calculate_expected_tdoas(self):
        """Calculate expected TDOAs for current target direction"""
        unit_direction = azimuth_elevation_to_unit_vector(self.target_azimuth, self.target_elevation)
        expected_lags = []
        
        # Calculate for all dynamically generated pairs
        for mic1_idx, mic2_idx in self.pair_indices:
            expected_lag = expected_pair_lag_samples(
                unit_direction,
                self.array_positions[mic1_idx],  # first mic
                self.array_positions[mic2_idx],  # second mic
                SAMPLE_RATE,
                float(SPEED_OF_SOUND_MPS)
            )
            expected_lags.append(expected_lag)
        
        return expected_lags
    
    def setup_srp_phat_grid(self):
        """Setup SRP-PHAT search grid with configurable resolution (az/el step in degrees)."""
        print("Setting up SRP-PHAT grid...")
        azimuth_range = tuple(SRP_AZ_RANGE)
        elevation_range = tuple(SRP_EL_RANGE)
        (
            self.az_grid,
            self.el_grid,
            self.steering_vectors,
            self.srp_power_map,
            summary,
        ) = setup_srp_phat_grid_context(
            azimuth_range_deg=azimuth_range,
            elevation_range_deg=elevation_range,
            az_step_deg=self.az_step_deg,
            el_step_deg=self.el_step_deg,
        )
        self.best_azimuth = 0.0
        self.best_elevation = 0.0
        self.best_power = 0.0
        print(summary)

    def _output_callback(self, outdata, frames, time_info, status):
        """sounddevice OutputStream callback: low-latency, never blocks UI.
        Fills stereo buffer from multi-channel audio_data based on self.audio_play_position.
        """
        try:
            # Default silence
            out = outdata
            # Quick exits
            if getattr(self, 'audio_data', None) is None or self.is_paused:
                out[:] = 0.0
                return
            start_idx = None
            with self._audio_lock:
                start_idx = int(self.audio_play_position)
            end_idx = start_idx + frames
            n_total = len(self.audio_data)
            if start_idx >= n_total:
                out[:] = 0.0
                return
            # Slice available region
            avail_end = min(end_idx, n_total)
            chunk = self.audio_data[start_idx:avail_end]
            # Mix to stereo: use first two selected channels if available; else average all to mono -> duplicate
            if chunk.ndim == 1:
                # mono -> duplicate
                left = right = chunk.astype(np.float32, copy=False)
            else:
                if chunk.shape[1] >= 2:
                    left = chunk[:, 0].astype(np.float32, copy=False)
                    right = chunk[:, 1].astype(np.float32, copy=False)
                else:
                    m = chunk.mean(axis=1).astype(np.float32, copy=False)
                    left = right = m
            # Prepare output buffer
            # Ensure C-contiguous assembly
            out_buf = np.empty((frames, 2), dtype=np.float32)
            n_have = avail_end - start_idx
            if n_have > 0:
                out_buf[:n_have, 0] = left
                out_buf[:n_have, 1] = right
            if n_have < frames:
                out_buf[n_have:, :] = 0.0
            # Apply volume
            out_buf *= 0.7
            # Write to provided buffer
            out[:] = out_buf
            # Advance playhead atomically
            with self._audio_lock:
                self.audio_play_position = start_idx + frames
        except Exception as e:
            # Never raise in callback
            try:
                # Fill silence on error
                outdata[:] = 0.0
            except Exception:
                pass

    def start_audio_stream(self):
        """Start audio playback using a persistent OutputStream (non-blocking)."""
        if not self.enable_audio or not AUDIO_AVAILABLE:
            return
            
        try:
            # If already running, nothing to do
            if self.audio_stream is not None:
                return
            # Sync audio position with visual position
            with self._audio_lock:
                self.audio_play_position = self.current_frame_idx * FRAME_SIZE
            print(f"Starting audio stream from position {self.audio_play_position}")
            # Open persistent output stream; uses our callback
            self.audio_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype='float32',
                blocksize=AUDIO_CHUNK_SIZE,
                device=self.playback_device,
                callback=self._output_callback,
            )
            self.audio_stream.start()
            print("✓ Audio stream started successfully")
                        
        except Exception as e:
            print(f"Could not start audio playback: {e}")
            self.enable_audio = False

    def stop_audio_stream(self):
        """Stop audio output stream safely."""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
            finally:
                self.audio_stream = None

    def setup_plots(self):
        self.fig = plt.figure(figsize=(24, 20))
        # Reorganized grid: 6 rows x 6 columns  
        # Row 0: Levels (spectrogram moved to a separate window for better visibility)
        # Rows 1-4: 4x3 GCC plots on left (cols 0-2), SRP-PHAT on right (cols 3-5) 
        # Row 5: Progress bar
        gs = self.fig.add_gridspec(6, 6, height_ratios=[1, 1.5, 1.5, 1.5, 1.5, 0.3], hspace=1.0, wspace=0.4)

        # GCC-PHAT plots for up to 12 pairs in a 4x3 grid on left side
        self.gcc_axes = []
        self.gcc_lines = []
        self.gcc_peak_lines = []
        self.gcc_tdoa_lines = []  # TDOA reference lines
        # Determine how many pairs to draw (use all for SRP, subset for visualization)
        pairs_to_show = min(len(self.pair_indices), 12)
        # Generate grid positions row=1..4, col=0..2
        gcc_positions = [(1 + (i // 3), i % 3) for i in range(pairs_to_show)]

        # Color palette for TDOA reference lines by reference mic index
        ref_colors = ['green', 'orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'blue']

        for i, (row, col) in enumerate(gcc_positions):
            ax = self.fig.add_subplot(gs[row, col])
            line, = ax.plot(self.lag_axis, self.gcc_data_pairs[i], 'b-', linewidth=2)
            peak_line = ax.axvline(x=0, color='r', linestyle='-', alpha=0.8, linewidth=2)
            
            # Color-code TDOA lines by reference mic (first index of the pair)
            m1, _ = self.pair_indices[i]
            tdoa_color = ref_colors[m1 % len(ref_colors)]
                
            tdoa_line = ax.axvline(x=0, color=tdoa_color, linestyle=':', alpha=0.8, linewidth=3, 
                                 label=f'Expected TDOA ({self.target_azimuth}°az, {self.target_elevation}°el)')
            
            ax.set_title(f'GCC-PHAT Pair {self.pair_labels[i]}', fontsize=11, fontweight='bold')
            ax.set_xlabel(f'Lag (samples @ {SAMPLE_RATE} Hz)', fontsize=9)
            ax.set_ylabel('Correlation', fontsize=9)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(-300, 300)
            ax.set_ylim(-0.1, 1.1)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            if i == 0:  # Only show legend on first plot
                ax.legend(loc='upper right', fontsize=9)
            
            self.gcc_axes.append(ax)
            self.gcc_lines.append(line)
            self.gcc_peak_lines.append(peak_line)
            self.gcc_tdoa_lines.append(tdoa_line)

        # SRP-PHAT Heatmap (right side, spanning rows 1-4 and columns 3-5 for wider display)
        if self.enable_srp_phat:
            self.srp_ax = self.fig.add_subplot(gs[1:5, 3:])
            if self.srp_power_map is not None:
                self.srp_im = self.srp_ax.imshow(self.srp_power_map, 
                                               aspect='auto', origin='lower', cmap='viridis',
                                               extent=[self.az_grid[0], self.az_grid[-1], 
                                                      self.el_grid[0], self.el_grid[-1]])
                # Keep a handle to the title Text so we can update it under blitting
                self.srp_title = self.srp_ax.set_title('SRP-PHAT Direction-of-Arrival', fontsize=12, fontweight='bold')
                try:
                    self.srp_title.set_animated(True)
                except Exception:
                    pass
                self.srp_ax.set_xlabel('Azimuth (degrees)', fontsize=10)
                self.srp_ax.set_ylabel('Elevation (degrees)', fontsize=10)
                
                # Add crosshair for best direction
                self.srp_az_line = self.srp_ax.axvline(x=self.best_azimuth, color='cyan', linewidth=2, alpha=0.8)
                self.srp_el_line = self.srp_ax.axhline(y=self.best_elevation, color='cyan', linewidth=2, alpha=0.8)
                
                # Add target direction reference if enabled
                if self.show_tdoa_reference:
                    self.srp_target_az = self.srp_ax.axvline(x=self.target_azimuth, color='green', 
                                                           linestyle=':', linewidth=3, alpha=0.8, label='Target Direction')
                    self.srp_target_el = self.srp_ax.axhline(y=self.target_elevation, color='green', 
                                                           linestyle=':', linewidth=3, alpha=0.8)
                    self.srp_ax.legend(loc='upper right', fontsize=9)
                
                # Colorbar
                self.srp_cbar = plt.colorbar(self.srp_im, ax=self.srp_ax, shrink=0.8)
                self.srp_cbar.set_label('SRP Power', rotation=270, labelpad=15)

                # Best DOA overlay (top-left inside axis) - blit-friendly
                best_str = f'Best: {self.best_azimuth:.1f}°az, {self.best_elevation:.1f}°el\nP: {self.best_power:.3f}'
                self.srp_best_text = self.srp_ax.text(
                    0.01, 0.99, best_str,
                    transform=self.srp_ax.transAxes,
                    ha='left', va='top', fontsize=10, color='white',
                    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                    animated=True
                )
                # Classifier overlay (bottom-right)
                self.cls_text = self.srp_ax.text(
                    0.99, 0.01,
                    'Classifier: disabled',
                    transform=self.srp_ax.transAxes,
                    ha='right', va='bottom', fontsize=10, color='white',
                    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                    animated=True
                )
                # Telemetry current-time marker (target symbol: outer ring + inner dot)
                if (self.telemetry is not None) and (self.audio_start_utc is not None) and (self.source_mode == SOURCE_WAV):
                    try:
                        # Outer ring (no fill), fixed point size
                        self.tele_marker_outer = self.srp_ax.scatter(
                            [], [], s=250, marker='o', facecolors='none', edgecolors='white', linewidths=2.0,
                            label='Telemetry', zorder=6)
                        # Inner dot (filled), fixed point size and bright color for visibility
                        self.tele_marker_inner = self.srp_ax.scatter(
                            [], [], s=40, marker='o', c='orangered', edgecolors='black', linewidths=0.5,
                            zorder=7)
                        # Add to legend if needed
                        handles, labels = self.srp_ax.get_legend_handles_labels()
                        if 'Telemetry' not in labels:
                            self.srp_ax.legend(loc='upper right', fontsize=9)
                    except Exception as e:
                        print(f"⚠️ Could not create telemetry marker: {e}")

        # Spectrogram moved to its own window for better visibility
        self.spec_fig = plt.figure(figsize=(12, 6))
        try:
            # Set a helpful window title if backend supports it
            self.spec_fig.canvas.manager.set_window_title('SRP-PHAT GCC Viewer - Spectrogram')
        except Exception:
            pass
        self.spec_ax0 = self.spec_fig.add_subplot(1, 1, 1)
        self.spec_ax0.set_title(f'Channel {self.mic_channels[0]} Spectrogram', fontsize=12, fontweight='bold')
        self.spec_ax0.set_xlabel('Time (frames)')
        self.spec_ax0.set_ylabel('Frequency (Hz)')
        # Use imshow with proper extent so Y axis is in Hz; we'll update extent per refresh
        init_img = np.zeros((len(self.spec_freqs), self.spec_frames_to_keep), dtype=np.float32)
        self.spec_im0 = self.spec_ax0.imshow(
            init_img,
            aspect='auto', origin='lower', cmap='viridis',
            extent=[0, self.spec_frames_to_keep, 0, SAMPLE_RATE/2]
        )
        # Add colorbar to display dynamic dB scale (vmin=vmax-60 .. vmax)
        try:
            self.spec_cbar = self.spec_fig.colorbar(self.spec_im0, ax=self.spec_ax0)
            self.spec_cbar.set_label('Spectrogram (dB, dynamic 60 dB range)')
        except Exception:
            pass
        nyq = SAMPLE_RATE / 2
        y_max_fixed = min(20000.0, float(nyq))
        ticks_fixed = np.array([0, 5000, 10000, 15000, 20000], dtype=float)
        ticks_fixed = ticks_fixed[ticks_fixed <= y_max_fixed]
        self.spec_ax0.set_yticks(ticks_fixed)
        self.spec_ax0.set_yticklabels([f'{int(t)}' for t in ticks_fixed])
        self.spec_ax0.set_ylim(0.0, y_max_fixed)

        # RMS Level Plot now spans the full top row in the main window
        self.level_ax = self.fig.add_subplot(gs[0, :])

        # RMS Level Plot
        self.level_line0, = self.level_ax.plot([], [], 'b-', label=f'Ch {self.mic_channels[0]}', linewidth=2)
        self.level_line1, = self.level_ax.plot([], [], 'r-', label=f'Ch {self.mic_channels[1]}', linewidth=2)
        self.level_line2, = self.level_ax.plot([], [], 'g-', label=f'Ch {self.mic_channels[2]}', linewidth=2)
        self.level_line3, = self.level_ax.plot([], [], 'm-', label=f'Ch {self.mic_channels[3]}', linewidth=2)
        self.level_line4, = self.level_ax.plot([], [], 'c-', label=f'Ch {self.mic_channels[4]}', linewidth=2)
        self.level_ax.set_title('Signal RMS Levels', fontsize=12, fontweight='bold')
        self.level_ax.set_ylabel('RMS Level')
        self.level_ax.set_xlabel('Frame')
        self.level_ax.legend()
        self.level_ax.grid(True, alpha=0.4)

        # Progress/status row (row 5, spanning all columns)
        self.progress_ax = self.fig.add_subplot(gs[5, :])
        if self.source_mode == SOURCE_WAV:
            self.progress_line, = self.progress_ax.plot([0, self.total_frames], [0, 0], 'k-', linewidth=8, alpha=0.3)
            self.progress_pos = self.progress_ax.axvline(x=0, color='red', linewidth=3)
            self.progress_ax.set_xlim(0, self.total_frames)
            self.progress_ax.set_ylim(-0.5, 0.5)
            self.progress_ax.set_xlabel(f'Frame (Total: {self.total_frames}, Duration: {self.total_frames*FRAME_SIZE/SAMPLE_RATE:.1f}s)')
            self.progress_ax.set_title('Playback Progress', fontsize=10)
            self.progress_ax.set_yticks([])
        else:
            # STREAM: show status text only
            self.progress_ax.axis('off')
            status_lines = [
                f"STREAM mode - device: {self.stream_device_name if self.stream_device_name else 'detecting...'}",
                f"Blocksize: {self.stream_blocksize} samples @ {SAMPLE_RATE} Hz",
            ]
            self.progress_text = self.progress_ax.text(
                0.01, 0.98, "\n".join(status_lines), transform=self.progress_ax.transAxes,
                ha='left', va='top', fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                animated=True
            )

        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle
        plt.suptitle(f'Multi-Mic GCC Viewer v{VERSION} - {os.path.basename(self.audio_file_path)}', fontsize=16)

        # Connect click event for seeking (WAV only)
        if self.source_mode == SOURCE_WAV:
            self.progress_ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add keyboard controls info with better visibility
        audio_status = "ON" if self.enable_audio else "OFF"
        bandpass_status = "ON" if self.use_bandpass_filter else "OFF"
        lowpass_status = "ON" if self.use_lowpass_filter else "OFF"
        self.control_text = self.fig.text(0.02, 0.02, 
                     f'Controls: SPACE=Play/Pause, R=Reset, A=Audio {audio_status}, Q=Quit, Click progress bar to seek\n'
                     f'TDOA Ref: T=Toggle, ←→=Azimuth ({self.target_azimuth:.0f}°), ↑↓=Elevation ({self.target_elevation:.0f}°)\n'
                     f'Filters: B=Bandpass {bandpass_status} (300-3400Hz), L=Lowpass {lowpass_status} (300Hz)', 
                     fontsize=9, fontweight='bold', color='white',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="darkblue", alpha=0.9, edgecolor="cyan", linewidth=2))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Display audio and telemetry start times for alignment (UTC-aware)
        audio_start_str = 'unknown'
        telemetry_start_str = 'unknown'
        if getattr(self, 'audio_start_utc', None) is not None:
            try:
                # Format without timezone offset suffix
                audio_start_str = self.audio_start_utc.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
            except Exception:
                audio_start_str = str(self.audio_start_utc)
        if getattr(self, 'telemetry', None) is not None and hasattr(self.telemetry, 'times') and len(self.telemetry.times) > 0:
            try:
                # Format without timezone offset suffix
                telemetry_start_str = self.telemetry.times[0].astimezone(timezone.utc).replace(tzinfo=None).isoformat()
            except Exception:
                telemetry_start_str = str(self.telemetry.times[0])
        # Place the time info inside the SRP axis so it participates in blitting safely
        if hasattr(self, 'srp_ax') and self.srp_ax is not None:
            self.time_info_text = self.srp_ax.text(
                0.99, 0.99,
                f'Audio start (UTC): {audio_start_str}\nTelemetry start (UTC): {telemetry_start_str}',
                transform=self.srp_ax.transAxes,
                ha='right', va='top', fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none'),
                animated=True
            )
        else:
            self.time_info_text = None

        # Telemetry readout (top-left, under best box): az/el/dist
        self.telemetry_info_text = None
        if self.enable_srp_phat and hasattr(self, 'srp_ax') and self.srp_ax is not None:
            self.telemetry_info_text = self.srp_ax.text(
                0.01, 0.85,
                'Drone: --.-° az --.-° el\nDistance: --- m',
                transform=self.srp_ax.transAxes,
                ha='left', va='top', fontsize=10, color='white',
                # Use transparent red background for telemetry overlay
                bbox=dict(facecolor='red', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'),
                animated=True
            )
        
        # Initialize TDOA reference lines
        self.update_tdoa_reference_lines()

    # Catalog resolution moved to src.catalog_utils.resolve_array_positions_from_catalog

    def update_tdoa_reference_lines(self):
        """Update the TDOA reference lines for current target direction"""
        if self.show_tdoa_reference:
            expected_lags = self.calculate_expected_tdoas()
            # Update only for the displayed subset of pairs
            for i in range(min(len(self.gcc_tdoa_lines), len(expected_lags))):
                expected_lag = expected_lags[i]
                self.gcc_tdoa_lines[i].set_xdata([expected_lag, expected_lag])
                self.gcc_tdoa_lines[i].set_visible(True)
                
            # Update legend label on first plot  
            self.gcc_tdoa_lines[0].set_label(f'Expected TDOA ({self.target_azimuth:.0f}°az, {self.target_elevation:.0f}°el)')
            self.gcc_axes[0].legend(loc='upper right', fontsize=9)
            
            # Update SRP-PHAT target reference lines
            if self.enable_srp_phat and hasattr(self, 'srp_target_az'):
                self.srp_target_az.set_xdata([self.target_azimuth, self.target_azimuth])
                self.srp_target_el.set_ydata([self.target_elevation, self.target_elevation])
                self.srp_target_az.set_visible(True)
                self.srp_target_el.set_visible(True)
        else:
            for tdoa_line in self.gcc_tdoa_lines:
                tdoa_line.set_visible(False)
            
            # Hide SRP-PHAT target lines too
            if self.enable_srp_phat and hasattr(self, 'srp_target_az'):
                self.srp_target_az.set_visible(False)
                self.srp_target_el.set_visible(False)

    def on_click(self, event):
        """Handle mouse clicks on progress bar for seeking"""
        if event.inaxes == self.progress_ax:
            self.current_frame_idx = max(0, min(int(event.xdata), self.total_frames - 1))
            
            # Sync audio position when seeking
            if self.enable_audio:
                with self._audio_lock:
                    self.audio_play_position = self.current_frame_idx * FRAME_SIZE
            # Reset pacing anchor so playback continues at real-time from the new position
            try:
                self._rt_anchor_time = time.time()
            except Exception:
                import time as _t
                self._rt_anchor_time = _t.time()
            self._rt_anchor_frame = self.current_frame_idx
            
            self.update_progress_display()
            print(f"Seeking to frame {self.current_frame_idx}")

    def on_key_press(self, event):
        """Handle keyboard controls"""
        if event.key == ' ':  # Space bar
            if self.is_paused:
                # Starting playback
                self.is_paused = False
                if self.enable_audio:
                    self.start_audio_stream()
                print("▶ Playback started")
                # Reset pacing anchor on resume so speed is real-time from current frame
                try:
                    self._rt_anchor_time = time.time()
                except Exception:
                    import time as _t
                    self._rt_anchor_time = _t.time()
                self._rt_anchor_frame = self.current_frame_idx
            else:
                # Pausing playback
                self.is_paused = True
                if self.enable_audio:
                    self.stop_audio_stream()
                print("⏸ Playback paused")
        elif event.key == 'r':  # Reset
            self.current_frame_idx = 0
            self.is_paused = True
            if self.enable_audio:
                self.stop_audio_stream()
                self.audio_play_position = 0
            print("⏮ Reset to beginning")
            # Clear pacing anchor (will re-anchor on resume)
            self._rt_anchor_time = None
            self._rt_anchor_frame = 0
        elif event.key == 'a':  # Toggle audio
            was_playing = not self.is_paused
            if was_playing:
                self.stop_audio_stream()
            
            self.enable_audio = not self.enable_audio
            
            if self.enable_audio and was_playing:
                self.start_audio_stream()
            
            self.update_control_text()
            print(f"🔊 Audio {'enabled' if self.enable_audio else 'disabled'}")
        elif event.key == 't':  # Toggle TDOA reference
            self.show_tdoa_reference = not self.show_tdoa_reference
            self.update_tdoa_reference_lines()
            self.fig.canvas.draw()
            print(f"🎯 TDOA reference {'enabled' if self.show_tdoa_reference else 'disabled'}")
        elif event.key == 'left':  # Decrease azimuth
            self.target_azimuth = (self.target_azimuth - 5) % 360
            self.update_tdoa_reference_lines()
            self.update_control_text()
            print(f"🧭 Azimuth: {self.target_azimuth:.0f}°")
        elif event.key == 'right':  # Increase azimuth
            self.target_azimuth = (self.target_azimuth + 5) % 360
            self.update_tdoa_reference_lines()
            self.update_control_text()
            print(f"🧭 Azimuth: {self.target_azimuth:.0f}°")
        elif event.key == 'up':  # Increase elevation
            self.target_elevation = min(90, self.target_elevation + 5)
            self.update_tdoa_reference_lines()
            self.update_control_text()
            print(f"📐 Elevation: {self.target_elevation:.0f}°")
        elif event.key == 'down':  # Decrease elevation
            self.target_elevation = max(-90, self.target_elevation - 5)
            self.update_tdoa_reference_lines()
            self.update_control_text()
            print(f"📐 Elevation: {self.target_elevation:.0f}°")
        elif event.key == 'b':  # Toggle bandpass filter
            self.use_bandpass_filter = not self.use_bandpass_filter
            filter_status = "ON" if self.use_bandpass_filter else "OFF"
            self.update_control_text()
            print(f"🎚️ Voice bandpass filter (300-3400 Hz): {filter_status}")
        elif event.key == 'l':  # Toggle low-pass filter
            self.use_lowpass_filter = not self.use_lowpass_filter
            filter_status = "ON" if self.use_lowpass_filter else "OFF"
            self.update_control_text()
            print(f"🎚️ Low-pass filter (300 Hz): {filter_status}")
        elif event.key == 'q':  # Quit
            if self.enable_audio:
                self.stop_audio_stream()
            print("Quitting...")
            plt.close('all')
    
    def update_control_text(self):
        """Update the control text display"""
        audio_status = "ON" if self.enable_audio else "OFF"
        bandpass_status = "ON" if self.use_bandpass_filter else "OFF"
        lowpass_status = "ON" if self.use_lowpass_filter else "OFF"
        self.control_text.set_text(
            f'Controls: SPACE=Play/Pause, R=Reset, A=Audio {audio_status}, Q=Quit, Click progress bar to seek\n'
            f'TDOA Ref: T=Toggle, ←→=Azimuth ({self.target_azimuth:.0f}°), ↑↓=Elevation ({self.target_elevation:.0f}°)\n'
            f'Filters: B=Bandpass {bandpass_status} (300-3400Hz), L=Lowpass {lowpass_status} (300Hz)'
        )
        # Make sure the styling is maintained
        self.control_text.set_fontweight('bold')
        self.control_text.set_color('white')
        self.fig.canvas.draw()

    def get_current_frame(self):
        """Get the current audio frame"""
        if self.source_mode == SOURCE_STREAM:
            # In STREAM mode, frames are fetched via queue
            return self._get_next_stream_frame()
        
        if self.current_frame_idx >= self.total_frames:
            return None
        
        start_idx = self.current_frame_idx * FRAME_SIZE
        end_idx = start_idx + FRAME_SIZE
        
        if end_idx > len(self.audio_data):
            # Pad with zeros if needed
            frame = np.zeros((FRAME_SIZE, len(self.mic_channels)))
            available_samples = len(self.audio_data) - start_idx
            if available_samples > 0:
                frame[:available_samples] = self.audio_data[start_idx:start_idx + available_samples]
        else:
            frame = self.audio_data[start_idx:end_idx]
        
        return frame

    # --- STREAM input plumbing ---
    def _print_audio_devices(self):
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, d in enumerate(devices):
                print(f"  {i}: {d['name']} (in: {d['max_input_channels']}, out: {d['max_output_channels']})")
        except Exception as e:
            print(f"Could not query audio devices: {e}")

    def _select_input_device(self):
        """Auto-select Zoom F8n Pro unless --device provided."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            raise RuntimeError(f"Audio device query failed: {e}")

        if self.input_device is not None:
            self.stream_device_name = devices[self.input_device]['name'] if 0 <= self.input_device < len(devices) else str(self.input_device)
            print(f"Using specified input device {self.input_device}: {self.stream_device_name}")
            return

        # Search for Zoom F8n Pro (case-insensitive substring)
        target = "zoom f8n pro"
        candidate_idx = None
        for i, d in enumerate(devices):
            name = str(d.get('name', ''))
            if target in name.lower() and d.get('max_input_channels', 0) >= 5:
                candidate_idx = i
                break

        # Fallback: first device with >= 5 input channels
        if candidate_idx is None:
            for i, d in enumerate(devices):
                if d.get('max_input_channels', 0) >= 5:
                    candidate_idx = i
                    break

        if candidate_idx is None:
            raise RuntimeError("No suitable input device found (need >= 5 input channels). Use --list-devices or --device.")

        self.input_device = candidate_idx
        self.stream_device_name = devices[candidate_idx]['name']
        print(f"Auto-selected input device {self.input_device}: {self.stream_device_name}")

    def _input_callback(self, indata, frames, time_info, status):
        if status and not getattr(status, 'input_overflow', False):
            print(f"Audio callback status: {status}")
        try:
            # Ensure float32
            block = indata.astype(np.float32, copy=False)
            # Select requested channels (by index) if available
            if block.shape[1] < len(self.mic_channels):
                # Not enough channels; push as-is and warn occasionally
                pass
            # Push to queue (drop-oldest policy)
            if self.input_queue.full():
                try:
                    _ = self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put_nowait(block.copy())
        except Exception:
            # Do not raise inside callback
            pass

    def _setup_input_stream_infrastructure(self):
        self.input_queue = queue.Queue(maxsize=5)
        try:
            self.input_stream = sd.InputStream(
                device=self.input_device,
                channels=max(self.mic_channels)+1,
                samplerate=SAMPLE_RATE,
                blocksize=self.stream_blocksize,
                dtype='float32',
                callback=self._input_callback,
            )
            self.input_stream.start()
            print(f"✓ Input stream started on device {self.input_device}: {self.stream_device_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to start input stream: {e}")

    def stop_input_stream(self):
        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
                print("Input stream stopped.")
        except Exception:
            pass

    def _get_next_stream_frame(self):
        if self.input_queue is None:
            return None
        try:
            block = self.input_queue.get_nowait()  # shape (frames, channels)
        except queue.Empty:
            return None
        # Ensure correct frame size
        if block.shape[0] != FRAME_SIZE:
            if block.shape[0] > FRAME_SIZE:
                block = block[:FRAME_SIZE, :]
            else:
                pad = np.zeros((FRAME_SIZE - block.shape[0], block.shape[1]), dtype=np.float32)
                block = np.vstack([block, pad])
        # Channel mapping
        if block.shape[1] <= max(self.mic_channels):
            # Not enough channels; return None to skip
            return None
        return block[:, self.mic_channels]

    def process_frame(self):
        """Process the current audio frame"""
        import time
        total_start = time.time()
        frame = self.get_current_frame()
        if frame is None:
            return False
        
        # Extract and filter all channels at once (shape: samples x channels)
        frame_ch = frame[:, :len(self.mic_channels)]
        # STREAM diagnostics: check raw activity of all input channels every ~1s
        if self.source_mode == SOURCE_STREAM and (self.current_frame_idx % 40 == 0):  # ~1s at 25ms frames
            try:
                # If the device actually provides more channels than selected, we can't see them here.
                # But we can still report activity for the selected channels.
                raw_std = [float(np.std(frame[:, i])) for i in range(frame.shape[1])]
                active = [(i, s) for i, s in enumerate(raw_std)]
                active.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join([f"ch{i} σ={s:.4g}" for i, s in active[:min(8, len(active))]])
                print(f"[STREAM] Channel activity (std): {top}")
                # Warn if any requested channels look silent
                silent = [i for i in range(len(self.mic_channels)) if raw_std[i] < 1e-6]
                if len(silent) > 0:
                    print(f"[STREAM][warn] Selected channels appear silent at indices: {silent}. Check --device and --channels mapping.")
            except Exception:
                pass
        t0 = time.time()
        frame_filt = self.apply_filter(frame_ch)  # (samples, channels)
        # Use ALL selected channels for GCC-PHAT
        mic_filt = [frame_filt[:, i] for i in range(frame_filt.shape[1])]
        filter_time = time.time() - t0

        # Debug signal levels + activity gate update (shared utility)
        rms_values = dsp_compute_levels(frame_filt, num_series=5)
        is_active = (max(rms_values) > self.srp_activity_rms_thresh)
        if is_active:
            self._srp_gate_counter = self.srp_gate_hold_frames
        else:
            self._srp_gate_counter = max(0, self._srp_gate_counter - 1)
        gate_allows = is_active or (self._srp_gate_counter > 0)
        self._srp_last_active = is_active
        if self.current_frame_idx % 100 == 0:  # Print every 100 frames
            print(f"Frame {self.current_frame_idx}: RMS: {rms_values} | active={is_active} hold={self._srp_gate_counter}")

        # GCC-PHAT for all dynamically generated pairs
        gcc_start = time.time()
        for pair_idx, (mic1_idx, mic2_idx) in enumerate(self.pair_indices):
            mic_stack = np.vstack([mic_filt[mic1_idx], mic_filt[mic2_idx]])
            gcc_result = compute_gcc_phat_singleblock(mic_stack, nfft=NFFT)
            self.gcc_data_pairs[pair_idx] = gcc_result[0]  # Only one pair per computation
        gcc_time = time.time() - gcc_start
        
        # Debug GCC-PHAT result for first pair
        if self.current_frame_idx % 100 == 0:
            gcc_max = np.max(self.gcc_data_pairs[0])
            gcc_peak_idx = np.argmax(self.gcc_data_pairs[0])
            print(f"GCC-PHAT Pair 0-1: Max={gcc_max:.4f}, Peak at index {gcc_peak_idx}")

        # RMS Levels for first 5 channels (display) via shared utility
        t_levels = time.time()
        first5 = dsp_compute_levels(frame_filt, num_series=5)
        self.level_history.append(first5)
        levels_time = time.time() - t_levels

        # Spectrogram data for channel 0 using shared utility
        t_spec = time.time()
        sig0 = mic_filt[0].astype(np.float32, copy=False)
        power_slice = dsp_compute_spec_power_slice(sig0, SAMPLE_RATE, SPEC_NFFT)
        self.spec_history_mic0.append(power_slice)
        spec_time = time.time() - t_spec

        # Feed classifier (non-blocking): mix to mono and enqueue
        try:
            if getattr(self, '_cls_enabled', False):
                # Force using the first selected channel for classifier input (pre-filter, to match training/CLI)
                mono48 = frame_ch[:, 0].astype(np.float32, copy=False)
                # Push a copy to avoid mutation
                self._cls_enqueue_audio(mono48.copy())
        except Exception:
            pass

        # SRP-PHAT computation (every N frames for performance)
        srp_time = 0.0
        srp_ran = False
        if self.enable_srp_phat:
            self.srp_frame_counter += 1
            if self.srp_frame_counter >= self.srp_update_interval:
                self.srp_frame_counter = 0
                try:
                    if gate_allows:
                        srp_start = time.time()
                        # Time GCC and SRP separately for precise DOA measurement
                        t0 = time.time()
                        gcc_curves = compute_gcc_phat_singleblock(frame_ch.T, nfft=NFFT)
                        t1 = time.time()
                        power_map, best_az, best_el, best_power = compute_srp_phat_windowed_max(
                            gcc_curves=gcc_curves,
                            mic_positions_m=self.array_positions,
                            pair_indices=self.pair_indices,
                            steering_unit_vectors=self.steering_vectors,
                            azimuth_grid_deg=self.az_grid,
                            elevation_grid_deg=self.el_grid,
                            sampling_rate_hz=SAMPLE_RATE,
                            speed_of_sound_mps=float(SPEED_OF_SOUND_MPS),
                            nfft=NFFT,
                            search_window=5,
                        )
                        t2 = time.time()
                        gcc_ms = (t1 - t0) * 1000.0
                        srp_ms = (t2 - t1) * 1000.0
                        doa_ms = (t2 - t0) * 1000.0
                        # Accumulate and print once per second
                        self._doa_log_frames += 1
                        self._gcc_ms_accum += gcc_ms
                        self._srp_ms_accum += srp_ms
                        self._doa_ms_accum += doa_ms
                        now_ts = time.time()
                        if self._doa_log_last_ts is None:
                            self._doa_log_last_ts = now_ts
                        if (now_ts - self._doa_log_last_ts) >= 1.0:
                            n = max(1, self._doa_log_frames)
                            print(
                                f"[PERF][doa] avg: gcc_ms={self._gcc_ms_accum/n:.2f} | srp_ms={self._srp_ms_accum/n:.2f} | doa_ms={self._doa_ms_accum/n:.2f}"
                            )
                            self._doa_log_last_ts = now_ts
                            self._doa_log_frames = 0
                            self._gcc_ms_accum = 0.0
                            self._srp_ms_accum = 0.0
                            self._doa_ms_accum = 0.0
                        srp_time = (t2 - srp_start) * 1000.0
                        if len(self.srp_power_history) > 0:
                            averaged_power_map = np.mean(list(self.srp_power_history), axis=0)
                            
                            # Find best direction from averaged map
                            best_idx = np.argmax(averaged_power_map)
                            num_az = len(self.az_grid)
                            el_idx, az_idx = best_idx // num_az, best_idx % num_az
                            avg_best_az = self.az_grid[az_idx]
                            avg_best_el = self.el_grid[el_idx]
                            avg_best_power = averaged_power_map.flat[best_idx]
                            
                            # Update SRP-PHAT results with averaged values
                            self.srp_power_map = averaged_power_map
                            self.best_azimuth = avg_best_az
                            self.best_elevation = avg_best_el
                            self.best_power = avg_best_power
                        else:
                            # Fallback to current frame if no history
                            self.srp_power_map = power_map
                            self.best_azimuth = best_az
                            self.best_elevation = best_el
                            self.best_power = best_power
                        
                        srp_time = time.time() - srp_start
                        srp_ran = True
                    else:
                        # Inactive: gently decay the map towards zero to avoid static
                        if self.srp_power_map is not None:
                            self.srp_power_map *= 0.9
                        srp_time = 0.0
                        srp_ran = False
                    # Debug output
                    if self.current_frame_idx % 100 == 0:
                        print(f"Averaged SRP-PHAT ({len(self.srp_power_history)} frames): Best direction {self.best_azimuth:.1f}°az, {self.best_elevation:.1f}°el, power: {self.best_power:.3f}")
                        
                except Exception as e:
                    print(f"SRP-PHAT computation error: {e}")

        # --- Performance summary for the whole frame ---
        total_time = time.time() - total_start
        if not hasattr(self, '_frame_perf_log_count'):
            self._frame_perf_log_count = 0
        # Log first few frames and then every 10 frames
        if self._frame_perf_log_count < 5 or (self.current_frame_idx % 10 == 0):
            # Track last SRP run time (ms) and compute amortized per-frame cost
            if srp_ran:
                self._last_srp_time_ms = srp_time * 1000.0
                self._last_srp_ran = True
            else:
                self._last_srp_ran = False

            amortized_srp_ms = (self._last_srp_time_ms / max(1, self.srp_update_interval))
            gcc_pct = (gcc_time / total_time * 100.0) if total_time > 0 else 0.0
            srp_pct_instant = ((srp_time / total_time) * 100.0) if total_time > 0 else 0.0
            srp_pct_amort = ((amortized_srp_ms/1000.0) / total_time * 100.0) if total_time > 0 else 0.0
            print(
                f"[PERF][proc] total={total_time*1000:.2f}ms | filter={filter_time*1000:.2f}ms | "
                f"gcc={gcc_time*1000:.2f}ms ({gcc_pct:.1f}%) | spec={spec_time*1000:.2f}ms | levels={levels_time*1000:.2f}ms | "
                f"srp_inst={srp_time*1000:.2f}ms ({srp_pct_instant:.1f}%) | srp_amort={amortized_srp_ms:.2f}ms ({srp_pct_amort:.1f}%) | "
                f"srp_ran={'yes' if srp_ran else 'no'}"
            )
            self._frame_perf_log_count += 1

        return True

    def update_progress_display(self):
        """Update the progress bar"""
        import time
        if self.source_mode == SOURCE_STREAM:
            # STREAM: update status text only (RTF/FPS updated elsewhere)
            if hasattr(self, 'progress_text') and self.progress_text is not None:
                self.progress_text.set_text(
                    f'STREAM mode - device: {self.stream_device_name} | blocksize={self.stream_blocksize} | idx={self.current_frame_idx}'
                )
            return
        
        self.progress_pos.set_xdata([self.current_frame_idx, self.current_frame_idx])
        current_time = self.current_frame_idx * FRAME_SIZE / SAMPLE_RATE
        total_time = self.total_frames * FRAME_SIZE / SAMPLE_RATE
        # Use a dedicated Text artist for progress that is safe for blitting
        progress_str = f'Frame: {self.current_frame_idx}/{self.total_frames} | Time: {current_time:.1f}s/{total_time:.1f}s'
        if not hasattr(self, 'progress_text') or self.progress_text is None:
            # Place the text INSIDE the axis (top-left) so tight_layout won't clip it
            self.progress_text = self.progress_ax.text(
                0.01, 0.98, progress_str,
                transform=self.progress_ax.transAxes,
                ha='left', va='top', fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                animated=True
            )
        else:
            self.progress_text.set_text(progress_str)

        # --- Real-time pacing measurement ---
        now = time.time()
        if self._rt_last_time is None:
            self._rt_last_time = now
            self._rt_last_frame = self.current_frame_idx
            return

        wall_dt = now - self._rt_last_time
        frames_adv = self.current_frame_idx - self._rt_last_frame
        if wall_dt > 0 and frames_adv >= 0:
            audio_dt = frames_adv * (FRAME_SIZE / SAMPLE_RATE)
            rtf = (audio_dt / wall_dt) if wall_dt > 0 else 0.0  # real-time factor
            fps = frames_adv / wall_dt if wall_dt > 0 else 0.0
            # update EMA
            if self._rt_ema is None:
                self._rt_ema = rtf
            else:
                a = self._rt_ema_alpha
                self._rt_ema = a * self._rt_ema + (1 - a) * rtf

            # Log first few and then every 50 frames
            if self._rt_log_count < 3 or (self.current_frame_idx % 50 == 0):
                print(
                    f"[PERF] Pace: wall={wall_dt*1000:.1f}ms | frames+={frames_adv} | audio={audio_dt*1000:.1f}ms | "
                    f"FPS={fps:.1f} | RTF={rtf:.2f} | RTF_EMA={self._rt_ema:.2f}"
                )
                self._rt_log_count += 1

            # move window
            self._rt_last_time = now
            self._rt_last_frame = self.current_frame_idx

    def update_plots(self, frame_num):
        """Animation update function"""
        import time
        cb_start = time.time()
        proc_ms = 0.0
        draw_ms = 0.0
        if not hasattr(self, '_cb_last_time'):
            self._cb_last_time = cb_start
            self._cb_log_count = 0

        # Real-time catch-up: optionally process only the latest frame to reduce compute
        if not self.is_paused and (self.source_mode == SOURCE_STREAM or self.current_frame_idx < self.total_frames):
            # Initialize pacing start when first entering running state
            if self._rt_anchor_time is None:
                self._rt_anchor_time = cb_start
                self._rt_anchor_frame = self.current_frame_idx

            # Desired frame index from wall clock relative to anchor
            hop_sec = FRAME_SIZE / SAMPLE_RATE
            elapsed = cb_start - self._rt_anchor_time
            desired_idx = self._rt_anchor_frame + int(elapsed / hop_sec)
            if self.skip_intermediate:
                # Jump to desired index and process only one frame to reduce compute
                # Clamp desired index within bounds
                if self.source_mode == SOURCE_WAV:
                    desired_idx = min(desired_idx, self.total_frames - 1)
                # Only move forward
                if desired_idx > self.current_frame_idx:
                    self.current_frame_idx = desired_idx
                p0 = time.time()
                if not self.process_frame():
                    return []
                self.current_frame_idx += 1
                proc_ms = (time.time() - p0) * 1000.0
            else:
                # Original behavior: process intermediate frames to catch up
                max_to_process = 20
                frames_to_process = max(0, min(desired_idx - self.current_frame_idx, max_to_process))
                processed = 0
                p0 = time.time()
                while processed < frames_to_process and (self.source_mode == SOURCE_STREAM or self.current_frame_idx < self.total_frames):
                    if not self.process_frame():
                        break
                    self.current_frame_idx += 1
                    processed += 1
                proc_ms = (time.time() - p0) * 1000.0
                # If no catch-up needed (callbacks are on time), process one frame to keep advancing
                if processed == 0 and (self.source_mode == SOURCE_STREAM or self.current_frame_idx < self.total_frames):
                    p0 = time.time()
                    if not self.process_frame():
                        return []
                    self.current_frame_idx += 1
                    proc_ms += (time.time() - p0) * 1000.0

        # Update GCC-PHAT plots for all 10 pairs (draw timing per section)
        t_draw_gcc = time.time()
        expected_lags = self.calculate_expected_tdoas() if self.show_tdoa_reference else [0]*10
        
        # Update lines and peaks
        for i in range(10):
            # Matplotlib expects NumPy arrays: convert backend array to NumPy
            try:
                y_np = asnumpy(self.gcc_data_pairs[i])
            except Exception:
                y_np = np.asarray(self.gcc_data_pairs[i])
            self.gcc_lines[i].set_ydata(y_np)
            # Use backend argmax and convert to Python int to avoid implicit conversions
            try:
                peak_idx_b = xp.argmax(self.gcc_data_pairs[i])
                peak_idx = int(asnumpy(peak_idx_b))
            except Exception:
                peak_idx = int(np.argmax(y_np))
            peak_lag = self.lag_axis[peak_idx]
            self.gcc_peak_lines[i].set_xdata([peak_lag, peak_lag])
        draw_gcc_lines_ms = (time.time() - t_draw_gcc) * 1000.0

        # Titles are expensive; update less frequently
        t_titles = time.time()
        if (self._cb_log_count < 5) or (self.current_frame_idx % 10 == 0):
            for i in range(10):
                # Backend-safe argmax and index
                try:
                    peak_idx_b = xp.argmax(self.gcc_data_pairs[i])
                    peak_idx = int(asnumpy(peak_idx_b))
                except Exception:
                    try:
                        peak_idx = int(np.argmax(asnumpy(self.gcc_data_pairs[i])))
                    except Exception:
                        peak_idx = int(np.argmax(self.gcc_data_pairs[i]))
                peak_lag = self.lag_axis[peak_idx]
                time_delay_us = (peak_lag / SAMPLE_RATE) * 1e6
                expected_delay_us = (expected_lags[i] / SAMPLE_RATE) * 1e6 if self.show_tdoa_reference else 0
                if self.show_tdoa_reference:
                    error_samples = peak_lag - expected_lags[i]
                    title = (f'GCC-PHAT {self.pair_labels[i]}\n'
                             f'Peak: {peak_lag:.1f} ({time_delay_us:.1f}μs)\n'
                             f'Expected: {expected_lags[i]:.1f} ({expected_delay_us:.1f}μs) | Δ: {error_samples:.1f}')
                else:
                    title = (f'GCC-PHAT {self.pair_labels[i]}\n'
                             f'Peak: {peak_lag:.1f} samples ({time_delay_us:.1f} μs)')
                self.gcc_axes[i].set_title(title, fontsize=8, fontweight='bold')
        draw_gcc_titles_ms = (time.time() - t_titles) * 1000.0

        # Update Spectrogram (separate window) using imshow
        t_draw_spec = time.time()
        if len(self.spec_history_mic0) > 0:
            spec_lin = np.array(list(self.spec_history_mic0)).T  # (n_freq, n_time)
            spec_db = 10 * np.log10(spec_lin + 1e-12)
            # Update image data and extent to current history length
            n_time = spec_db.shape[1]
            self.spec_im0.set_data(spec_db)
            self.spec_im0.set_extent([0, n_time, 0, SAMPLE_RATE/2])
            # Dynamic color scaling (like wavorstream.py)
            vmax = float(np.max(spec_db)) if np.isfinite(np.max(spec_db)) else 0.0
            vmin = vmax - 60.0
            self.spec_im0.set_clim(vmin=vmin, vmax=vmax)
            # Fixed Y-limits: 0 to 20 kHz (or Nyquist if lower)
            nyq = SAMPLE_RATE / 2
            y_max_fixed = min(20000.0, float(nyq))
            self.spec_ax0.set_ylim(0.0, y_max_fixed)
            # Request redraw of the spectrogram figure (since it's not managed by the main blit)
            try:
                self.spec_fig.canvas.draw_idle()
            except Exception:
                pass
        draw_spec_ms = (time.time() - t_draw_spec) * 1000.0

        # Update RMS Levels
        t_draw_levels = time.time()
        if len(self.level_history) > 0:
            levels = np.array(list(self.level_history))
            x_data = np.arange(len(levels))
            self.level_line0.set_data(x_data, levels[:, 0])
            self.level_line1.set_data(x_data, levels[:, 1])
            self.level_line2.set_data(x_data, levels[:, 2])
            self.level_line3.set_data(x_data, levels[:, 3])
            self.level_line4.set_data(x_data, levels[:, 4])
            self.level_ax.relim()
            self.level_ax.autoscale_view()
        draw_levels_ms = (time.time() - t_draw_levels) * 1000.0

        # Update SRP-PHAT heatmap with simple normalization
        t_draw_srp = time.time()
        if self.enable_srp_phat and hasattr(self, 'srp_im'):
            power_map = self.srp_power_map.copy()
            
            # Static suppression: if inactive or very low contrast, show near-zero map
            power_min = power_map.min()
            power_max = power_map.max()
            contrast = power_max - power_min
            if (not self._srp_last_active) or (contrast < 1e-6) or (self.best_power < 1e-6):
                enhanced_map = np.zeros_like(power_map)
            else:
                # Normalize to [0,1] range and apply gamma correction
                normalized_map = (power_map - power_min) / max(1e-12, contrast)
                enhanced_map = np.power(normalized_map, self.gamma_correction)
            
            # Update heatmap data
            self.srp_im.set_data(enhanced_map)
            self.srp_im.set_clim(vmin=0, vmax=1)  # Fixed range for consistency
            
            # Update crosshair for best direction
            self.srp_az_line.set_xdata([self.best_azimuth, self.best_azimuth])
            self.srp_el_line.set_ydata([self.best_elevation, self.best_elevation])
            
            # Update overlay best text
            if hasattr(self, 'srp_best_text') and self.srp_best_text is not None:
                self.srp_best_text.set_text(
                    f'Best: {self.best_azimuth:.1f}°az, {self.best_elevation:.1f}°el\nP: {self.best_power:.3f}'
                )
            # Update telemetry marker and readout (non-realtime only)
            if (self.telemetry is not None) and (self.audio_start_utc is not None) and (self.source_mode == SOURCE_WAV):
                try:
                    current_time_sec = self.current_frame_idx * FRAME_SIZE / SAMPLE_RATE
                    t_utc = self.audio_start_utc + timedelta(seconds=current_time_sec)
                    az, el = self.telemetry.azel_at(t_utc, self.array_lat, self.array_lon, self.array_alt)
                    # Compute distance using ENU vector magnitude
                    lat, lon, alt = self.telemetry.position_at(t_utc)
                    x, y, z = geodetic_to_ecef(lat, lon, alt)
                    e, n, u = ecef_to_enu(x, y, z, self.array_lat, self.array_lon, self.array_alt)
                    dist_m = float((e**2 + n**2 + u**2) ** 0.5)
                    # Update telemetry readout text regardless of marker presence
                    if hasattr(self, 'telemetry_info_text') and self.telemetry_info_text is not None:
                        self.telemetry_info_text.set_text(
                            f'Drone: {az:.1f}° az {el:.1f}° el\nDistance: {dist_m:.0f} m'
                        )
                    # Optionally update marker if created
                    if (self.tele_marker_outer is not None) and (self.tele_marker_inner is not None):
                        # Clamp to bounds of SRP grid extents so the marker remains visible
                        az = float(az)
                        el = float(el)
                        az_min, az_max = self.az_grid[0], self.az_grid[-1]
                        el_min, el_max = self.el_grid[0], self.el_grid[-1]
                        orig_el = el
                        orig_az = az
                        if az < az_min or az > az_max or el < el_min or el > el_max:
                            az = (az % 360.0)
                            az = max(az_min, min(az_max, az))
                            el = max(el_min, min(el_max, el))
                            if abs(el - el_min) < 1e-6:
                                el = el_min + 0.5
                            if self.current_frame_idx % 200 == 0:
                                print(f"[tele] out-of-range az/el ({orig_az:.2f},{orig_el:.2f}) -> clamped to ({az:.2f},{el:.2f})")
                        self.tele_marker_outer.set_offsets([[az, el]])
                        self.tele_marker_inner.set_offsets([[az, el]])
                except Exception:
                    # Fallback text and hide marker if it exists
                    if hasattr(self, 'telemetry_info_text') and self.telemetry_info_text is not None:
                        self.telemetry_info_text.set_text('Drone: --.-° az --.-° el\nDistance: --- m')
                    if (self.tele_marker_outer is not None) and (self.tele_marker_inner is not None):
                        self.tele_marker_outer.set_offsets([[np.nan, np.nan]])
                        self.tele_marker_inner.set_offsets([[np.nan, np.nan]])
            # Update classifier overlay text if available
            try:
                if hasattr(self, 'cls_text') and self.cls_text is not None:
                    txt = self._cls_get_overlay_text()
                    if txt is not None:
                        self.cls_text.set_text(txt)
            except Exception:
                pass
        draw_srp_ms = (time.time() - t_draw_srp) * 1000.0

        # Update progress
        d0 = time.time()
        self.update_progress_display()
        draw_ms = (time.time() - d0) * 1000.0

        # Callback pacing log with draw breakdown
        cb_end = time.time()
        cb_dt = (cb_end - self._cb_last_time) * 1000.0
        if self._cb_log_count < 3 or (self.current_frame_idx % 50 == 0):
            print(
                f"[PERF][draw] interval={cb_dt:.1f}ms | proc={proc_ms:.2f}ms | "
                f"gcc_lines={draw_gcc_lines_ms:.2f}ms | gcc_titles={draw_gcc_titles_ms:.2f}ms | "
                f"spec={draw_spec_ms:.2f}ms | levels={draw_levels_ms:.2f}ms | srp={draw_srp_ms:.2f}ms | "
                f"progress={draw_ms:.2f}ms | idx={self.current_frame_idx}"
            )
            self._cb_log_count += 1
        self._cb_last_time = cb_end

        # Return list excludes spectrogram artist (it belongs to a separate figure)
        return_list = (self.gcc_lines + self.gcc_peak_lines + self.gcc_tdoa_lines + 
                      [self.level_line0, self.level_line1, self.level_line2, self.level_line3, self.level_line4])
        if self.source_mode == SOURCE_WAV:
            return_list += [self.progress_pos]
        # Include the progress text artist for blitting updates
        if hasattr(self, 'progress_text') and self.progress_text is not None:
            return_list += [self.progress_text]
        # Include static time info box for blitting
        if hasattr(self, 'time_info_text') and self.time_info_text is not None:
            return_list += [self.time_info_text]
        
        # Add SRP-PHAT elements to return list if available
        if self.enable_srp_phat and hasattr(self, 'srp_im'):
            return_list += [self.srp_im, self.srp_az_line, self.srp_el_line]
            if hasattr(self, 'srp_title') and self.srp_title is not None:
                return_list += [self.srp_title]
            if hasattr(self, 'srp_best_text') and self.srp_best_text is not None:
                return_list += [self.srp_best_text]
            if self.show_tdoa_reference:
                return_list += [self.srp_target_az, self.srp_target_el]
            if (self.tele_marker_outer is not None) and (self.tele_marker_inner is not None):
                return_list += [self.tele_marker_outer, self.tele_marker_inner]
            if hasattr(self, 'telemetry_info_text') and self.telemetry_info_text is not None:
                return_list += [self.telemetry_info_text]
            if hasattr(self, 'cls_text') and self.cls_text is not None:
                return_list += [self.cls_text]
        
        return return_list

    # ===== Classifier integration (optional) =====
    def _init_classifier(self):
        """Initialize classifier models if available. Non-fatal on failure."""
        self._cls_enabled = False
        # Prefer env MODELS_ROOT; otherwise use config default
        self._cls_models_root = os.environ.get('MODELS_ROOT', CONFIG_MODELS_ROOT)
        self._cls_paths = {
            'x': os.path.join(self._cls_models_root, 'classifier_x_binary.h5'),
            'y': os.path.join(self._cls_models_root, 'classifier_y_make.h5'),
        }
        # Discover Z submodels
        z_root = os.path.join(self._cls_models_root, 'classifier_z_')
        self._cls_z_models: Dict[str, Dict[str, str]] = {}
        try:
            root = Path(self._cls_models_root)
            if root.exists():
                for p in root.glob('classifier_z_*'):
                    if p.is_dir():
                        make = p.name.replace('classifier_z_', '')
                        model_path = str(p / 'model.h5')
                        labels_path = str(p / 'labels.txt')
                        if os.path.exists(model_path) and os.path.exists(labels_path):
                            self._cls_z_models[make] = {'model': model_path, 'labels': labels_path}
        except Exception:
            pass

        # Try loading models if TF is available
        self._cls_models: Dict[str, Any] = {}
        self._cls_labels: Dict[str, list[str]] = {}
        # Y (make) labels loaded from models root if present
        self._cls_y_labels: list[str] = []
        if TF_AVAILABLE:
            try:
                if os.path.exists(self._cls_paths['x']):
                    self._cls_models['x'] = tf.keras.models.load_model(self._cls_paths['x'])
                if os.path.exists(self._cls_paths['y']):
                    self._cls_models['y'] = tf.keras.models.load_model(self._cls_paths['y'])
                    # Attempt to load Y labels alongside the Y model
                    try:
                        y_labels_path = os.path.join(self._cls_models_root, 'classifier_y_make_labels.txt')
                        if os.path.exists(y_labels_path):
                            def _parse_label_line(line: str) -> str | None:
                                s = line.strip()
                                if not s:
                                    return None
                                # Try split on arrow if present
                                if '→' in s:
                                    parts = s.split('→', 1)
                                    s = parts[1].strip() if len(parts) > 1 else s
                                # Remove any leading index and separators like '1:', '1-', '1,'
                                s = re.sub(r'^\s*\d+\s*[:\-.,]*\s*', '', s)
                                return s if s else None
                            with open(y_labels_path, 'r', encoding='utf-8') as f:
                                labels = []
                                for ln in f:
                                    lbl = _parse_label_line(ln)
                                    if lbl:
                                        labels.append(lbl)
                                self._cls_y_labels = labels
                            if len(self._cls_y_labels) > 0:
                                print(f"[cls] Loaded {len(self._cls_y_labels)} Y labels from {y_labels_path}")
                    except Exception as e:
                        print(f"[cls] warn: could not load Y labels: {e}")
                for mk, paths in self._cls_z_models.items():
                    try:
                        self._cls_models[f'z:{mk}'] = tf.keras.models.load_model(paths['model'])
                        # load labels
                        with open(paths['labels'], 'r') as f:
                            self._cls_labels[mk] = [ln.strip() for ln in f if ln.strip()]
                    except Exception as e:
                        print(f"[cls] warn: could not load Z model for {mk}: {e}")
                if 'x' in self._cls_models:
                    self._cls_enabled = True
                    print(f"✓ Classifier enabled (models root: {self._cls_models_root})")
                else:
                    print("[cls] X model not found; classifier disabled")
            except Exception as e:
                print(f"[cls] model load failed: {e}")
                self._cls_enabled = False
        else:
            print("[cls] TensorFlow missing; classifier disabled")

        # Runtime buffers/state
        # Align classifier sample rate with training pipeline (48 kHz)
        self._cls_sr_target = 48000
        self._cls_win_s = 0.2  # 200 ms frames
        self._cls_seq_steps = 10  # 2 s per sequence
        self._cls_frame_buf_48k: list[np.ndarray] = []  # accumulate 48k chunks
        # Rolling buffer of MFCC frames (each is a 40-dim vector); we will form sequences from the last 10 frames
        self._cls_seq_mfcc: deque = deque(maxlen=self._cls_seq_steps)
        self._cls_last: Optional[Dict[str, Any]] = None
        # 18 s aggregation per paper: keep last 3 sequence decisions
        self._cls_hist: deque = deque(maxlen=3)
        # Debug toggle via env: CLS_DEBUG=1
        try:
            self._cls_debug = (os.environ.get('CLS_DEBUG', '0') not in ('0', 'false', 'False'))
        except Exception:
            self._cls_debug = False
        # Allow overriding which index is 'Drone' when X outputs 2 logits/probs
        try:
            self._cls_x_drone_idx = int(os.environ.get('CLS_X_DRONE_INDEX', '1'))
        except Exception:
            self._cls_x_drone_idx = 1
        # Async worker
        self._cls_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)
        self._cls_stop = threading.Event()
        if self._cls_enabled:
            t = threading.Thread(target=self._cls_worker_loop, name='ClassifierWorker', daemon=True)
            t.start()

    def _cls_enqueue_audio(self, mono48: np.ndarray):
        try:
            if not self._cls_enabled:
                return
            if self._cls_q.full():
                _ = self._cls_q.get_nowait()
            self._cls_q.put_nowait(mono48)
        except Exception:
            pass

    def _cls_worker_loop(self):
        hop48 = FRAME_SIZE  # samples per viewer frame at 48k
        target_len_44k = int(round(self._cls_win_s * self._cls_sr_target))
        acc_44: list[np.ndarray] = []
        acc_len = 0
        while not self._cls_stop.is_set():
            try:
                chunk48 = self._cls_q.get(timeout=0.05)
            except queue.Empty:
                continue
            # Resample this chunk to 44.1k
            try:
                # Resample to 48 kHz to match training
                y44 = librosa.resample(chunk48, orig_sr=SAMPLE_RATE, target_sr=self._cls_sr_target)
            except Exception:
                # fallback: skip chunk
                continue
            acc_44.append(y44)
            acc_len += len(y44)
            # Process in 200 ms windows
            while acc_len >= target_len_44k:
                # Concatenate if multiple segments
                buf = np.concatenate(acc_44) if len(acc_44) > 1 else acc_44[0]
                win = buf[:target_len_44k]
                rest = buf[target_len_44k:]
                acc_44 = [rest] if len(rest) > 0 else []
                acc_len = len(rest)
                # Preprocess window to MFCC frames (T, 40) without averaging
                mfcc_frames = self._cls_make_mfcc(win, self._cls_sr_target)  # shape (T,40)
                if mfcc_frames is None:
                    continue
                # Append each frame to rolling buffer
                if mfcc_frames.ndim == 1:
                    self._cls_seq_mfcc.append(mfcc_frames.astype(np.float32))
                else:
                    for i in range(mfcc_frames.shape[0]):
                        self._cls_seq_mfcc.append(mfcc_frames[i].astype(np.float32))
                # When we have 10 frames, form a sequence from the last 10 frames
                if len(self._cls_seq_mfcc) >= self._cls_seq_steps:
                    # Use exactly the last 10 frames to form (10,40)
                    last10 = list(self._cls_seq_mfcc)[-self._cls_seq_steps:]
                    seq = np.stack(last10, axis=0)  # (10, 40)
                    self._cls_run_inference(seq)

    def _cls_make_mfcc(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        try:
            # Peak normalization per 200 ms frame
            mx = float(np.max(np.abs(y)))
            if mx > 0:
                y = (y / mx).astype(np.float32)
            # Match training MFCC settings: n_fft=2048, hop_length=512, fmax=8000, center=False
            n_fft = 2048
            hop = 512
            # Match training settings in src/droneprint/features.py which use librosa defaults (center=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, fmax=8000.0, n_fft=n_fft, hop_length=hop, center=True)
            # Return per-frame MFCCs as (T,40) with L2 rescale per frame (row)
            if mfcc.ndim == 1:
                v = mfcc.astype(np.float32).reshape(1, -1)
            else:
                v = mfcc.T.astype(np.float32)  # (T,40)
            # L2 rescale each frame
            for i in range(v.shape[0]):
                nrm = float(np.linalg.norm(v[i]))
                if nrm > 0:
                    v[i] = (v[i] / nrm).astype(np.float32)
            return v
        except Exception:
            return None

    def _cls_run_inference(self, seq_10x40: np.ndarray):
        if not self._cls_enabled:
            return
        try:
            x = seq_10x40.astype(np.float32)
            # Try common input shapes
            preds_x = None
            m_x = self._cls_models.get('x')
            if m_x is not None:
                preds_x = self._cls_predict_with_model(m_x, x)
            result: Dict[str, Any] = {
                'drone_prob': None,
                'drone_label': None,
                'make_label': None,
                'make_probs': None,
                'model_label': None,
                'model_probs': None,
            }
            if preds_x is not None:
                # Assume binary [non, drone] or scalar sigmoid
                if preds_x.ndim == 0:
                    pr = float(preds_x)
                elif preds_x.shape[-1] == 1:
                    pr = float(preds_x[..., 0])
                elif preds_x.shape[-1] == 2:
                    # For 2-class softmax, decide class via argmax and compute p_drone via configured index
                    idx = int(self._cls_x_drone_idx)
                    if idx < 0 or idx > 1:
                        idx = 1
                    # Ensure numpy array for consistent ops
                    vec_np = np.array(preds_x).astype(float)
                    # Compute predicted class by argmax
                    pred_idx = int(np.argmax(vec_np))
                    # Probability of the 'Drone' class as configured
                    pr = float(vec_np[..., idx])
                    if getattr(self, '_cls_debug', False):
                        try:
                            vec = vec_np.tolist()
                            print(f"[cls][Xvec] {vec} using idx={idx} pred_idx={pred_idx}")
                        except Exception:
                            pass
                    # Set label from argmax rather than thresholding a fixed index
                    result['drone_label'] = 'Drone' if pred_idx == idx else 'Non-Drone'
                else:
                    pr = float(preds_x.max())
                # Store probability (for display/averaging) and ensure label set if not overridden above
                result['drone_prob'] = pr
                if result.get('drone_label') is None:
                    result['drone_label'] = 'Drone' if pr >= 0.5 else 'Non-Drone'
            # If drone, run Y (make)
            if (result['drone_prob'] is not None) and (result['drone_prob'] >= 0.5) and ('y' in self._cls_models):
                m_y = self._cls_models['y']
                probs_y = self._cls_predict_with_model(m_y, x)
                if probs_y is not None and probs_y.size > 0:
                    mk_idx = int(np.argmax(probs_y))
                    result['make_probs'] = probs_y.tolist()
                    # Map make index to label using loaded Y labels if available
                    try:
                        if hasattr(self, '_cls_y_labels') and (mk_idx < len(self._cls_y_labels)):
                            result['make_label'] = self._cls_y_labels[mk_idx]
                        else:
                            result['make_label'] = f"MAKE_{mk_idx}"
                    except Exception:
                        result['make_label'] = f"MAKE_{mk_idx}"
                    # Try Z model for this make if present
                    mk_key = None
                    # Prefer exact/case-insensitive match to the predicted make label
                    try:
                        predicted_make = result.get('make_label') or ''
                        pm_norm = (predicted_make or '').strip().lower()
                        if pm_norm:
                            for k in self._cls_z_models.keys():
                                if k.strip().lower() == pm_norm:
                                    mk_key = k
                                    break
                        # Fallback heuristic if no exact match: choose the only Z model if it's the only one
                        if mk_key is None and len(self._cls_z_models) == 1:
                            mk_key = list(self._cls_z_models.keys())[0]
                    except Exception:
                        pass
                    if mk_key and (f'z:{mk_key}' in self._cls_models):
                        m_z = self._cls_models[f'z:{mk_key}']
                        probs_z = self._cls_predict_with_model(m_z, x)
                        if probs_z is not None and probs_z.size > 0:
                            md_idx = int(np.argmax(probs_z))
                            labels = self._cls_labels.get(mk_key)
                            lbl = labels[md_idx] if labels and md_idx < len(labels) else f"MODEL_{md_idx}"
                            result['model_label'] = lbl
                            result['model_probs'] = probs_z.tolist()
            # Update last and history for 18s aggregation
            self._cls_last = result
            try:
                self._cls_hist.append({'drone_prob': result.get('drone_prob')})
            except Exception:
                pass
            # Debug print
            if self._cls_debug:
                dp = result.get('drone_prob')
                mk = result.get('make_label')
                md = result.get('model_label')
                print(f"[cls] X={dp:.3f} | Y={mk or '-'} | Z={md or '-'}")
        except Exception as e:
            print(f"[cls] inference error: {e}")

    def _cls_predict_with_model(self, model, x10x40: np.ndarray) -> Optional[np.ndarray]:
        """Format input based on model.input_shape; log feature stats when debug is enabled.
        Tries common layouts: (B,10,40), (B,10,40,1), (B,40,10), (B,40,10,1). Returns squeezed np.ndarray or None.
        """
        try:
            B = 1
            x = x10x40
            # Debug stats on features
            if getattr(self, '_cls_debug', False):
                try:
                    vmin = float(x.min()); vmax = float(x.max()); vmean = float(x.mean()); vnorm = float(np.linalg.norm(x))
                    print(f"[cls][feat] min={vmin:.3f} max={vmax:.3f} mean={vmean:.3f} L2={vnorm:.3f}")
                except Exception:
                    pass
            ish = getattr(model, 'input_shape', None)
            # Build candidate inputs
            candidates = []
            # (B, 10, 40)
            candidates.append(x[None, ...])
            # (B, 10, 40, 1)
            candidates.append(x[None, ..., None])
            # (B, 40, 10)
            candidates.append(np.transpose(x, (1, 0))[None, ...])
            # (B, 40, 10, 1)
            candidates.append(np.transpose(x, (1, 0))[None, ..., None])
            # If input_shape present, prioritize matching ranks/shapes loosely
            if ish is not None:
                rank = len([d for d in ish if d is not None]) if isinstance(ish, (list, tuple)) else None
                # Simple heuristic: try to pick candidate with matching rank
                ordered = []
                for arr in candidates:
                    if rank is None or arr.ndim == rank:
                        ordered.insert(0, arr)
                    else:
                        ordered.append(arr)
                candidates = ordered
            # Try candidates
            for inp in candidates:
                try:
                    # Debug: log model/input shapes
                    if getattr(self, '_cls_debug', False):
                        try:
                            print(f"[cls][predict] input shape tried: {inp.shape} model.input_shape={getattr(model, 'input_shape', None)}")
                        except Exception:
                            pass
                    p = model.predict(inp, verbose=0)
                    if getattr(self, '_cls_debug', False):
                        try:
                            arr = np.array(p).squeeze()
                            print(f"[cls][pred] shape={arr.shape} min={float(arr.min()):.3f} max={float(arr.max()):.3f} mean={float(arr.mean()):.3f}")
                        except Exception:
                            pass
                    return np.array(p).squeeze()
                except Exception:
                    continue
            return None
        except Exception as e:
            if getattr(self, '_cls_debug', False):
                print(f"[cls][predict] error: {e}")
            return None

    def _cls_get_overlay_text(self) -> Optional[str]:
        if not getattr(self, '_cls_enabled', False):
            return 'Classifier: disabled'
        r = self._cls_last
        if not r:
            return 'Classifier: warming up…'
        # Compute 18s average over last 3 sequences if available
        avg18 = None
        try:
            if len(self._cls_hist) > 0:
                vals = [float(h.get('drone_prob')) for h in self._cls_hist if h.get('drone_prob') is not None]
                if len(vals) > 0:
                    avg18 = float(np.mean(vals))
        except Exception:
            avg18 = None
        parts = []
        if r.get('drone_prob') is not None:
            if avg18 is not None:
                parts.append(f"X: {r['drone_label']} ({r['drone_prob']:.2f}, avg18s {avg18:.2f})")
            else:
                parts.append(f"X: {r['drone_label']} ({r['drone_prob']:.2f})")
        if r.get('make_label'):
            parts.append(f"Y: {r['make_label']}")
        if r.get('model_label'):
            parts.append(f"Z: {r['model_label']}")
        return "\n".join(parts) if parts else 'Classifier: no decision'

    def start_viewer(self):
        """Start the file-based viewer"""
        print(f"\n🎬 Starting Multi-Mic GCC Viewer v{VERSION}...")
        print('Controls: SPACE=Play/Pause, R=Reset, A=Toggle Audio, Q=Quit, Click progress bar to seek')
        print('TDOA Reference: T=Toggle, ←→=Adjust Azimuth, ↑↓=Adjust Elevation')
        print(f'🎯 Initial target: {self.target_azimuth:.0f}° azimuth, {self.target_elevation:.0f}° elevation')
        
        # Audio setup info
        if self.enable_audio:
            print('🔊 Threaded audio playback enabled - press SPACE to start')
        else:
            print("🔇 Audio is disabled")
        
        # Start paused
        self.is_paused = True
        
        # Create animation paced by target display FPS (e.g., 10-15 FPS)
        interval_ms = int(max(1, 1000.0 / max(1e-6, self.display_fps)))
        # Note: blit=True for faster redraws (we return the changed artists)
        ani = animation.FuncAnimation(
            self.fig, self.update_plots,
            interval=interval_ms,
            blit=True,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        finally:
            # Cleanup audio
            if self.enable_audio:
                print("Cleaning up audio...")
                self.stop_audio_stream()
            # Cleanup input stream if STREAM mode
            if self.source_mode == SOURCE_STREAM:
                self.stop_input_stream()
        
        return ani

    def _resolve_telemetry_from_catalog(self, wav_path: str) -> str | None:
        """Given a WAV file path, look up its catalog recording doc and choose the best telemetry CSV by overlap.
        Search catalog/recordings/*.json for a doc whose 'path' matches wav_path (relative or absolute).
        Return the telemetry CSV path (as stored in the doc), or None if not found.
        """
        # __file__ lives at the project root alongside 'catalog/' and 'field_recordings/'
        repo_root = Path(__file__).resolve().parent  # project root
        cat_rec_dir = repo_root / 'catalog' / 'recordings'
        if not cat_rec_dir.exists():
            return None
        # Normalize candidates for comparison (relative to repo root and absolute)
        wav_abs = Path(wav_path).resolve()
        try:
            wav_rel = str(wav_abs.relative_to(repo_root))
        except Exception:
            wav_rel = None
        best_path = None
        best_overlap = -1.0
        rec_doc = None
        for rec_file in cat_rec_dir.glob('*.json'):
            try:
                with open(rec_file, 'r') as f:
                    rec = json.load(f)
                rec_path = rec.get('path')
                if not rec_path:
                    continue
                # Match by relative or absolute
                if rec_path == str(wav_abs) or (wav_rel and rec_path == wav_rel):
                    rec_doc = rec
                    overlaps = rec.get('overlaps') or []
                    for ov in overlaps:
                        ov_s = float(ov.get('overlap_s') or 0.0)
                        t_path = ov.get('telemetry_path')
                        if t_path and ov_s >= 0 and ov_s > best_overlap:
                            best_overlap = ov_s
                            best_path = t_path
                    break
            except Exception:
                continue
        if best_path:
            # Resolve to absolute path if stored relative
            p = Path(best_path)
            if not p.is_absolute():
                p = repo_root / best_path
            return str(p)
        # Fallback: choose telemetry by time-range containment/nearest if overlaps are empty
        # Determine audio start UTC
        def _parse_iso(ts: str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                return None
        audio_start = None
        if rec_doc is not None:
            audio_start = _parse_iso(rec_doc.get('start_utc') or '')
        if audio_start is None and TELEMETRY_AVAILABLE:
            try:
                audio_start = get_wav_start_utc(str(wav_abs), getattr(self, 'audio_local_utc_offset', '+00:00'))
            except Exception:
                audio_start = None
        if audio_start is None:
            return None
        # Scan telemetry docs
        cat_tel_dir = repo_root / 'catalog' / 'telemetry'
        if not cat_tel_dir.exists():
            return None
        chosen = None
        chosen_dist = None
        for tel_file in cat_tel_dir.glob('*.json'):
            try:
                with open(tel_file, 'r') as f:
                    tel = json.load(f)
                t_start = _parse_iso(tel.get('start_utc') or '')
                t_end = _parse_iso(tel.get('end_utc') or '')
                t_path = tel.get('path')
                if not t_path or not t_start or not t_end:
                    continue
                if t_start <= audio_start <= t_end:
                    chosen = t_path
                    chosen_dist = 0.0
                    break
                # distance to nearest boundary
                if audio_start < t_start:
                    dist = (t_start - audio_start).total_seconds()
                else:
                    dist = (audio_start - t_end).total_seconds()
                if (chosen is None) or (dist < (chosen_dist or float('inf'))):
                    chosen = t_path
                    chosen_dist = dist
            except Exception:
                continue
        if chosen:
            p = Path(chosen)
            if not p.is_absolute():
                p = repo_root / chosen
            return str(p)
        return None

def main():
    parser = argparse.ArgumentParser(description='Multi-microphone GCC-PHAT viewer with audio playback.')
    parser.add_argument('audio_file', nargs='?', default='', help='Path to audio file (WAV, FLAC, MP3, etc.)')
    parser.add_argument('--channels', type=str, default='', 
                       help='Comma-separated channel indices to use (default: all from array geometry)')
    parser.add_argument('--speed', type=float, default=1.0, 
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--array-heading', type=float, default=0.0,
                       help='Array heading in degrees clockwise from North (default: 0.0)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio playback (visualization only)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Disable bandpass filter (may help with GCC-PHAT)')
    parser.add_argument('--playback-device', type=int, default=None,
                        help='ID of the audio device to use for playback.')
    parser.add_argument('--source', type=str, choices=[SOURCE_WAV, SOURCE_STREAM], default=SOURCE_WAV,
                        help='Audio source mode: WAV (default) or STREAM (live input).')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device ID for STREAM mode. If not provided, auto-select Zoom F8n Pro or first device with >=5 inputs.')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices and exit.')
    parser.add_argument('--array-ref', type=str, default=None,
                        help='Array ID from catalog/arrays/array_defs.json (required in STREAM mode).')
    # Display performance controls
    parser.add_argument('--fps', type=float, default=VIEWER_FPS_DEFAULT, help='Target display FPS (reduce to ~10-15 to save compute).')
    parser.add_argument('--no-skip-intermediate', dest='skip_intermediate', action='store_false',
                        help='Process all intermediate frames instead of skipping to the latest.')
    # SRP grid resolution controls
    parser.add_argument('--az-step', type=float, default=SRP_AZ_STEP_DEG_DEFAULT, help='Azimuth step in degrees for SRP grid (default from config).')
    parser.add_argument('--el-step', type=float, default=SRP_EL_STEP_DEG_DEFAULT, help='Elevation step in degrees for SRP grid (default from config).')
    # Telemetry options (non-realtime overlay)
    parser.add_argument('--telemetry', type=str, default=None, help='Path to telemetry CSV file (UTC in time_iso). Overrides auto-selection if provided.')
    parser.add_argument('--telemetry-auto', action='store_true', default=True,
                        help='Auto-select telemetry CSV from catalog overlaps when available (default: on).')
    parser.add_argument('--no-telemetry-auto', action='store_false', dest='telemetry_auto',
                        help='Disable auto-selection of telemetry from catalog.')
    parser.add_argument('--array-lat', type=float, default=DEFAULT_ARRAY_LAT, help='Array latitude in degrees (negative for South).')
    parser.add_argument('--array-lon', type=float, default=DEFAULT_ARRAY_LON, help='Array longitude in degrees (positive for East).')
    parser.add_argument('--array-alt', type=float, default=DEFAULT_ARRAY_ALT, help='Array altitude in meters ASL.')
    parser.add_argument('--audio-local-utc-offset', type=str, default='+10:00', help='Local timezone offset for WAV OriginationTime (e.g., +10:00 for Brisbane).')
    parser.set_defaults(skip_intermediate=True)
    args = parser.parse_args()

    try:
        if args.channels.strip():
            mic_channels = list(map(int, args.channels.split(',')))
        else:
            mic_channels = None  # Use geometry length as default
        if mic_channels is not None and len(mic_channels) < 5:
            raise ValueError('Please specify at least five channels.')
        
        if args.source == SOURCE_WAV:
            if not args.audio_file or not os.path.exists(args.audio_file):
                raise ValueError(f'Audio file not found: {args.audio_file}')
        else:
            # STREAM mode validation: require array selection
            if not args.array_ref:
                raise ValueError('STREAM mode requires --array-ref <ARRAY_ID> from catalog/arrays/array_defs.json')
        
        enable_audio = not args.no_audio
        if args.list_devices and (not AUDIO_AVAILABLE):
            print('sounddevice is not available; cannot list devices.')
            return 1
        viewer = SRPGCCViewer(
            args.audio_file,
            mic_channels=mic_channels,
            playback_speed=args.speed,
            enable_audio=enable_audio,
            playback_device=args.playback_device,
            array_heading_deg=args.array_heading,
            source_mode=args.source,
            input_device=args.device,
            list_devices=args.list_devices,
            telemetry_path=args.telemetry,
            array_lat=args.array_lat,
            array_lon=args.array_lon,
            array_alt=args.array_alt,
            audio_local_utc_offset=args.audio_local_utc_offset,
            telemetry_auto=args.telemetry_auto,
            display_fps=args.fps,
            skip_intermediate=args.skip_intermediate,
            az_step_deg=args.az_step, el_step_deg=args.el_step,
            array_ref=args.array_ref
        )
        
        # Disable filtering if requested
        if args.no_filter:
            print("⚠️ Bandpass filter disabled")
            viewer.apply_filter = lambda x: x  # No-op filter
        
        ani = viewer.start_viewer()
        
    except Exception as e:
        print(f'Error: {e}')
        return 1

    return 0

if __name__ == '__main__':
    exit(main())