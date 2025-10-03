from __future__ import annotations

from src.messages import AudioFrame, Detection
from src.dsp_utils import compute_levels
from src.config import SRP_ACTIVITY_RMS_THRESH, FRAME_SIZE, SAMPLE_RATE, SRP_GATE_HOLD_SEC


class DroneDetector:
    """Simple activity detector based on per-channel RMS with hold-time gating."""
    def __init__(self, thresh: float = SRP_ACTIVITY_RMS_THRESH, hold_sec: float = SRP_GATE_HOLD_SEC):
        self.thresh = float(thresh)
        hop_sec = FRAME_SIZE / float(SAMPLE_RATE)
        self.hold_frames = max(1, int(hold_sec / hop_sec))
        self._counter = 0

    def process(self, frame: AudioFrame) -> Detection:
        levels = compute_levels(frame.audio, num_series=5)
        active_now = (max(levels) > self.thresh)
        if active_now:
            self._counter = self.hold_frames
        else:
            self._counter = max(0, self._counter - 1)
        gate = active_now or (self._counter > 0)
        return Detection(
            active=gate,
            scores={"rms_max": float(max(levels))},
            thresholds={"rms": float(self.thresh)},
        )
