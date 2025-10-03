"""
Central configuration for SRP-PHAT workbench.

Exports constants used across viewers and scripts.
"""
from pathlib import Path

# --- Core sampling and frame configuration ---
SAMPLE_RATE: int = 48000
FRAME_SIZE: int = 1200          # 25 ms @ 48 kHz
NFFT: int = 1200                 # Match frame size to avoid zero padding issues

# --- UI update pacing ---
UPDATE_INTERVAL: int = 50        # ms between animation updates

# --- Spectrogram configuration ---
SPEC_DISPLAY_SECONDS: int = 5    # history window (seconds)
SPEC_NFFT: int = 512             # ~93.75 Hz bins @ 48 kHz; Hann window via scipy.spectrogram

# --- Filtering bands ---
LOWPASS_FREQ: int = 8000         # optional low-pass
HIGHPASS_FREQ: int = 100         # optional high-pass

# --- Audio playback chunking ---
AUDIO_CHUNK_SIZE: int = 4800     # 100 ms @ 48 kHz for smoother output buffering

# --- Physical constants ---
SPEED_OF_SOUND_MPS: float = 343.0

# --- SRP/Viewer default tunables ---
SRP_ENABLED_DEFAULT: bool = True
SRP_UPDATE_INTERVAL: int = 2
SRP_GAMMA: float = 1.3
SRP_AVG_FRAMES: int = 2
SRP_ACTIVITY_RMS_THRESH: float = 2e-4
SRP_GATE_HOLD_SEC: float = 0.5

# SRP search grid defaults
SRP_AZ_STEP_DEG_DEFAULT: float = 1.0
SRP_EL_STEP_DEG_DEFAULT: float = 1.0
SRP_AZ_RANGE: tuple[float, float] = (0.0, 360.0)
SRP_EL_RANGE: tuple[float, float] = (0.0, 90.0)

# Viewer pacing defaults
VIEWER_FPS_DEFAULT: float = 15.0
SKIP_INTERMEDIATE_DEFAULT: bool = True

# --- Fallback array site coordinates (used only if not provided by CLI/catalog) ---
DEFAULT_ARRAY_LAT: float = -26.696758333
DEFAULT_ARRAY_LON: float = 152.885445
DEFAULT_ARRAY_ALT: float = 0.0

# --- Models configuration ---
# Default root containing classifier models. This resolves to
#   <repo>/src/droneprint/models/models_smoke
# You can override at runtime with the environment variable MODELS_ROOT
# or via application-specific CLI flags if available.
MODELS_ROOT: str = str((Path(__file__).resolve().parent / 'droneprint' / 'models' / 'models_smoke'))
