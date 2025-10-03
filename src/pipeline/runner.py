from __future__ import annotations

import time
from typing import Optional
from src.messages import FrameResult
from src.config import SAMPLE_RATE, SPEC_NFFT
from src.dsp_utils import compute_levels, compute_spec_power_slice


class PipelineRunner:
    def __init__(self, source, detector, doa, classifier, emitter, target_fps: float = 15.0, total_frames: Optional[int] = None):
        self.source = source
        self.detector = detector
        self.doa = doa
        self.classifier = classifier
        self.emitter = emitter
        self.target_hop = 1.0 / float(target_fps)
        self.total_frames = total_frames
        # Preserve last classification between frames
        self._last_cls = None
        # Telemetry for pacing visibility
        self._log_last_ts = None
        self._log_frames = 0
        self._log_overruns = 0
        self._log_accum = {
            "detection_ms": 0.0,
            "doa_ms": 0.0,
            "classification_ms": 0.0,
            "features_ms": 0.0,  # levels + spectrogram
            "emit_ms": 0.0,
            "total_compute_ms": 0.0,
            "spare_ms": 0.0,
        }

    def run(self, max_frames: Optional[int] = None):
        # Prepare DOA engine (steering vectors, grids)
        self.doa.prepare()
        if self.total_frames is not None and hasattr(self.emitter, "set_total_frames"):
            self.emitter.set_total_frames(self.total_frames)

        t0 = time.perf_counter()
        for frame in self.source.start():
            loop_start = time.perf_counter()
            # Detection
            t = time.perf_counter(); det = self.detector.process(frame); t_det = (time.perf_counter() - t)
            # DOA estimate
            t = time.perf_counter(); doa_res = self.doa.process(frame, det.active); t_doa = (time.perf_counter() - t)
            # Classification (optional)
            if self.classifier:
                t = time.perf_counter()
                pre = self.classifier.preprocess(frame, doa_res)
                cls = self.classifier.predict(pre)
                # If classifier is throttled (or disabled) and returns None, reuse last classification
                if cls is None:
                    cls = self._last_cls
                else:
                    self._last_cls = cls
                t_cls = (time.perf_counter() - t)
            else:
                cls = None; t_cls = 0.0
            # Features (levels + spectrogram)
            t = time.perf_counter(); levels = compute_levels(frame.audio, num_series=5); spec_slice = compute_spec_power_slice(frame.audio[:, 0], SAMPLE_RATE, SPEC_NFFT); t_feats = (time.perf_counter() - t)
            result = FrameResult(
                frame_idx=frame.idx,
                detection=det,
                doa=doa_res,
                classification=cls,
                meta={"sr": frame.sr, "frame_size": frame.audio.shape[0]},
                levels_first5=levels,
                spec_power_mic0=spec_slice.tolist(),
            )
            # Emit
            t = time.perf_counter(); self.emitter.emit(result); t_emit = (time.perf_counter() - t)

            # Pace to target FPS accounting for processing time (anchor to t0 to avoid drift)
            now = time.perf_counter()
            next_due = t0 + (frame.idx + 1) * self.target_hop
            sleep_s = max(0.0, next_due - now)
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                self._log_overruns += 1

            # Accumulate logging stats
            total_compute = (time.perf_counter() - loop_start) - max(0.0, sleep_s)
            self._log_frames += 1
            self._log_accum["detection_ms"] += (t_det * 1000.0)
            self._log_accum["doa_ms"] += (t_doa * 1000.0)
            self._log_accum["classification_ms"] += (t_cls * 1000.0)
            self._log_accum["features_ms"] += (t_feats * 1000.0)
            self._log_accum["emit_ms"] += (t_emit * 1000.0)
            self._log_accum["total_compute_ms"] += (total_compute * 1000.0)
            self._log_accum["spare_ms"] += (sleep_s * 1000.0)

            now_wall = time.perf_counter()
            if self._log_last_ts is None:
                self._log_last_ts = now_wall
            if (now_wall - self._log_last_ts) >= 1.0:
                sec = now_wall - self._log_last_ts
                tx_fps = self._log_frames / sec if sec > 0 else 0.0
                avg = {k: (v / max(1, self._log_frames)) for (k, v) in self._log_accum.items()}
                hop_ms = self.target_hop * 1000.0
                rtf = (avg["total_compute_ms"] / hop_ms) if hop_ms > 0 else 0.0
                ontime = max(0, self._log_frames - self._log_overruns)
                overrun_pct = (100.0 * self._log_overruns / max(1, self._log_frames))
                print(
                    f"[perf] tx_fps={tx_fps:.2f} | hop_s={self.target_hop:.4f} ({hop_ms:.1f} ms) | rtf={rtf:.3f} | "
                    f"compute_ms total={avg['total_compute_ms']:.2f} (det={avg['detection_ms']:.2f}, doa={avg['doa_ms']:.2f}, classif={avg['classification_ms']:.2f}, features={avg['features_ms']:.2f}, emit={avg['emit_ms']:.2f}) | "
                    f"spare_ms={avg['spare_ms']:.2f} | ontime={ontime}/{self._log_frames} | overruns={self._log_overruns} ({overrun_pct:.1f}%)"
                )
                # reset window
                self._log_last_ts = now_wall
                self._log_frames = 0
                self._log_overruns = 0
                for k in self._log_accum.keys():
                    self._log_accum[k] = 0.0
            if max_frames is not None and frame.idx >= max_frames:
                break
