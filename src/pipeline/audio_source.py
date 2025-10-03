from __future__ import annotations

import time
import numpy as np
from typing import Generator, List, Optional
import queue
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False
from src.audio_io import load_audio
from src.config import SAMPLE_RATE, FRAME_SIZE
from src.messages import AudioFrame


class AudioSource:
    """Minimal file-based audio source yielding frames as AudioFrame.
    Future: add ALSA/stream backend with sounddevice for low-latency capture.
    """
    def __init__(self, backend: str = "file", file_path: Optional[str] = None, channels: Optional[List[int]] = None, device: Optional[int | str] = None):
        self.backend = backend
        self.file_path = file_path
        self.channels = channels
        self.device = device
        if self.backend not in ("file", "stream"):
            raise NotImplementedError("AudioSource backend must be 'file' or 'stream'")

    @staticmethod
    def list_devices() -> list:
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is not available")
        return sd.query_devices()

    @staticmethod
    def autoselect_input(min_channels: int = 5, preferred_name_substr: str = "zoom f8n pro") -> int:
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is not available")
        try:
            devices = sd.query_devices()
        except Exception as e:
            raise RuntimeError(f"Audio device query failed: {e}")
        # Prefer named device
        cand = None
        for i, d in enumerate(devices):
            name = str(d.get('name', ''))
            if preferred_name_substr.lower() in name.lower() and d.get('max_input_channels', 0) >= min_channels:
                cand = i
                break
        # Fallback: first with enough inputs
        if cand is None:
            for i, d in enumerate(devices):
                if d.get('max_input_channels', 0) >= min_channels:
                    cand = i
                    break
        if cand is None:
            raise RuntimeError(f"No suitable input device found (need >={min_channels} input channels). Use --list-devices or --device.")
        return cand

    def start(self) -> Generator[AudioFrame, None, None]:
        if self.backend == "file":
            assert self.file_path, "file_path is required for file backend"
            audio, sr = load_audio(self.file_path, SAMPLE_RATE)  # (samples, channels)
            if self.channels:
                if max(self.channels) >= audio.shape[1]:
                    raise ValueError(f"Requested channel {max(self.channels)} but file has {audio.shape[1]} channels")
                audio = audio[:, self.channels]
            total = len(audio)
            idx = 0
            frame_idx = 0
            start_time = time.time()
            while idx < total:
                end = min(idx + FRAME_SIZE, total)
                frame = np.zeros((FRAME_SIZE, audio.shape[1]), dtype=np.float32)
                frame[: (end - idx)] = audio[idx:end, :]
                ts = time.time() - start_time
                yield AudioFrame(idx=frame_idx, ts=ts, audio=frame, sr=SAMPLE_RATE)
                frame_idx += 1
                idx = end
            return
        # STREAM backend via ALSA/sounddevice
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is not available. Install it to use stream backend.")
        q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        channels = len(self.channels) if self.channels else None

        def callback(indata, frames, time_info, status):
            if status:
                # drop frames on over/under-run but keep running
                pass
            buf = indata.copy().astype(np.float32, copy=False)
            # Channel mapping if requested
            if self.channels is not None:
                if max(self.channels) >= buf.shape[1]:
                    return  # skip if not enough channels
                buf = buf[:, self.channels]
            try:
                q.put_nowait(buf)
            except queue.Full:
                # drop oldest to keep latency bounded
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(buf)
                except Exception:
                    pass

        # Determine device index/name
        sel_device = self.device
        if sel_device is None:
            try:
                sel_device = AudioSource.autoselect_input(min_channels=(len(self.channels) if self.channels else 5))
            except Exception as e:
                raise RuntimeError(f"Could not auto-select input device: {e}")

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=channels,
            dtype='float32',
            blocksize=FRAME_SIZE,
            latency='low',
            device=sel_device,
            callback=callback,
        )
        frame_idx = 0
        start_time = time.time()
        with stream:
            while True:
                buf = q.get()
                # Ensure exact FRAME_SIZE x channels (pad/truncate)
                if buf.shape[0] < FRAME_SIZE:
                    out = np.zeros((FRAME_SIZE, buf.shape[1]), dtype=np.float32)
                    out[:buf.shape[0]] = buf
                else:
                    out = buf[:FRAME_SIZE]
                ts = time.time() - start_time
                yield AudioFrame(idx=frame_idx, ts=ts, audio=out, sr=SAMPLE_RATE)
                frame_idx += 1
