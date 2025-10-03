from __future__ import annotations

import json
import urllib.request
from typing import Optional
from datetime import datetime, timezone, timedelta
import time
import threading
import queue
import gzip
from io import BytesIO

from src.config import SAMPLE_RATE, FRAME_SIZE, SPEC_NFFT, SPEC_DISPLAY_SECONDS, SRP_GAMMA
import os
from src.messages import FrameResult


class Emitter:
    def __init__(self, mode: str = "print", webhook_url: Optional[str] = None, normalize: bool = True, post_timeout: float = 0.5, source_mode: str = "WAV", ws_url: Optional[str] = None):
        self.mode = mode  # "print" or "webhook"
        self.webhook_url = webhook_url.rstrip("/") if webhook_url else None
        self.normalize = normalize
        self.post_timeout = float(post_timeout)
        self.source_mode = str(source_mode).upper()  # "WAV" or "STREAM"
        self.ws_url = None
        if ws_url:
            # normalize ws(s) url and path: allow user to pass base host or full path
            u = ws_url.strip()
            if u.endswith('/'):
                u = u[:-1]
            # If user passed just ws://host:port, append /ws_ingest
            if not u.endswith('/ws_ingest'):
                u = u + '/ws_ingest'
            self.ws_url = u
        self._init_sent = False
        self._az_grid = None
        self._el_grid = None
        self._total_frames = None
        self._post_fail_count = 0
        # Optional time base for UTC alignment (especially for WAV replay)
        self._start_utc: Optional[datetime] = None
        # WS client handle (lazy)
        self._ws = None
        # Wall-clock pacing anchors (for UTC aligned to real time)
        self._wall_anchor_perf: Optional[float] = None
        self._utc_anchor: Optional[datetime] = None
        # Async network worker
        self._tx_queue: "queue.Queue[dict]" = queue.Queue(maxsize=8)
        self._tx_thread: Optional[threading.Thread] = None
        self._tx_stop = threading.Event()
        self._drops = 0
        self._last_warn_ts = 0.0
        # Throttle console prints in print mode: print every N frames (default 15). Always print first 3 frames.
        try:
            self._print_every = max(1, int(os.environ.get('PRINT_FRAME_EVERY', '15')))
        except Exception:
            self._print_every = 15
        self._print_count = 0

    def set_grids(self, az_grid, el_grid):
        self._az_grid = az_grid
        self._el_grid = el_grid

    def set_total_frames(self, n: Optional[int]):
        self._total_frames = n

    def set_time_base(self, start_utc: Optional[datetime]):
        """Provide a UTC time base (first-sample time). Frames will be timestamped
        as start_utc + frame_idx * (FRAME_SIZE / SAMPLE_RATE). If None, current UTC is used."""
        self._start_utc = start_utc

    def _http_post_json(self, path: str, payload: dict) -> None:
        assert self.webhook_url, "webhook_url is not set"
        url = f"{self.webhook_url}{path}"
        data = json.dumps(payload).encode('utf-8')
        # Attempt gzip compression to reduce payload size
        try:
            buf = BytesIO()
            with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
                gz.write(data)
            gz_data = buf.getvalue()
            if len(gz_data) < len(data) * 0.9:  # use gzip only if it meaningfully helps
                data = gz_data
                headers = {
                    'Content-Type': 'application/json',
                    'Content-Encoding': 'gzip',
                }
            else:
                headers = {'Content-Type': 'application/json'}
        except Exception:
            headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.post_timeout) as resp:
                _ = resp.read()
        except Exception as e:
            # Do not block pipeline on network issues; warn occasionally
            self._post_fail_count += 1
            if self._post_fail_count <= 3 or (self._post_fail_count % 50 == 0):
                print(f"[webhook][warn] POST {url} failed ({e}); continuing without blocking (fail #{self._post_fail_count})")

    # --- WebSocket helpers ---
    def _ws_connect(self):
        if not self.ws_url:
            return
        if self._ws is not None:
            return
        try:
            import websocket  # websocket-client
            # Use a short timeout to avoid blocking pipeline
            self._ws = websocket.create_connection(self.ws_url, timeout=self.post_timeout)
            try:
                # Ensure subsequent send/recv also honor short timeouts
                self._ws.settimeout(self.post_timeout)
            except Exception:
                pass
        except Exception as e:
            self._ws = None
            # degrade silently after a few notices
            self._post_fail_count += 1
            if self._post_fail_count <= 3 or (self._post_fail_count % 50 == 0):
                print(f"[ws][warn] connect {self.ws_url} failed ({e}); continuing without blocking (fail #{self._post_fail_count})")

    def _ws_send_json(self, message: dict) -> None:
        if not self.ws_url:
            return
        if self._ws is None:
            self._ws_connect()
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps(message))
        except Exception as e:
            # try one reconnect
            try:
                if self._ws is not None:
                    self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._post_fail_count += 1
            if self._post_fail_count <= 3 or (self._post_fail_count % 50 == 0):
                print(f"[ws][warn] send failed ({e}); will retry connect next frame (fail #{self._post_fail_count})")

    # --- Async worker ---
    def _ensure_worker(self):
        if self.mode == "print":
            return  # no worker needed
        if self._tx_thread is None or not self._tx_thread.is_alive():
            self._tx_stop.clear()
            self._tx_thread = threading.Thread(target=self._worker_loop, name="EmitterTx", daemon=True)
            self._tx_thread.start()

    def _worker_loop(self):
        last_frame_sent = None
        while not self._tx_stop.is_set():
            try:
                item = self._tx_queue.get(timeout=0.05)
            except queue.Empty:
                item = None
            # Coalesce: if multiple frames queued, keep pulling to send only the latest frame payload
            if item is not None and item.get("type") == "frame":
                last_frame_sent = item  # candidate
                # drain quickly any additional frame payloads to coalesce
                while True:
                    try:
                        nxt = self._tx_queue.get_nowait()
                        if nxt.get("type") == "frame":
                            last_frame_sent = nxt
                        else:
                            # send non-frame immediately after
                            self._dispatch(nxt)
                    except queue.Empty:
                        break
                # finally send the latest frame
                self._dispatch(last_frame_sent)
                last_frame_sent = None
            elif item is not None:
                self._dispatch(item)

    def _dispatch(self, item: dict):
        try:
            if item.get("type") == "init":
                if self.ws_url:
                    self._ws_send_json({"type": "init", "payload": item["payload"]})
                elif self.webhook_url:
                    self._http_post_json("/init", item["payload"])
            elif item.get("type") == "frame":
                if self.ws_url:
                    self._ws_send_json({"type": "frame", "payload": item["payload"]})
                elif self.webhook_url:
                    self._http_post_json("/frame", item["payload"])
        except Exception:
            # errors already logged in lower-level methods
            pass

    def _emit_init(self):
        payload = {
            "version": "1.0",
            "sample_rate_hz": int(SAMPLE_RATE),
            "frame_size": int(FRAME_SIZE),
            "spectrogram": {
                "nfft": int(SPEC_NFFT),
                # compatibility key used by earlier receiver
                "spec_nfft": int(SPEC_NFFT),
                "display_seconds": float(SPEC_DISPLAY_SECONDS),
            },
            "srp": {
                "az_grid_deg": list(map(float, self._az_grid)) if self._az_grid is not None else [],
                "el_grid_deg": list(map(float, self._el_grid)) if self._el_grid is not None else [],
                "gamma": float(SRP_GAMMA),
            },
            "progress": {
                "total_frames": int(self._total_frames) if self._total_frames is not None else None,
                "source_mode": self.source_mode,
                # Use time base if available to reflect recording start
                "utc_iso": (self._start_utc.isoformat() if isinstance(self._start_utc, datetime) else datetime.now(timezone.utc).isoformat())
            },
        }
        if self.mode == "print":
            az = len(payload["srp"]["az_grid_deg"]) if payload["srp"]["az_grid_deg"] else 0
            el = len(payload["srp"]["el_grid_deg"]) if payload["srp"]["el_grid_deg"] else 0
            print(
                f"[INIT] sr={payload['sample_rate_hz']} frame={payload['frame_size']} spec_nfft={payload['spectrogram']['nfft']} disp_s={payload['spectrogram']['display_seconds']} "
                f"srp_grid=az {az} x el {el} source={payload['progress']['source_mode']} utc={payload['progress']['utc_iso']} total_frames={payload['progress']['total_frames']}"
            )
        else:
            self._ensure_worker()
            # Priority enqueue: clear any old init and push latest
            try:
                self._tx_queue.put_nowait({"type": "init", "payload": payload})
            except queue.Full:
                # Drop oldest and retry once
                try:
                    _ = self._tx_queue.get_nowait()
                    self._tx_queue.put_nowait({"type": "init", "payload": payload})
                except Exception:
                    self._drops += 1
        self._init_sent = True
        # Set wall-clock pacing anchors if we have a recording UTC start
        if isinstance(self._start_utc, datetime):
            self._wall_anchor_perf = time.perf_counter()
            self._utc_anchor = self._start_utc

    def emit(self, result: FrameResult):
        # Ensure init sent once grids are available
        if not self._init_sent and (result.doa.az_grid_deg is not None) and (result.doa.el_grid_deg is not None):
            self.set_grids(result.doa.az_grid_deg, result.doa.el_grid_deg)
            self._emit_init()

        # Build frame payload similar to webhook_emitter
        best = {
            "azimuth_deg": float(result.doa.azimuth_deg),
            "elevation_deg": float(result.doa.elevation_deg),
            "power": float(result.doa.power),
        }
        srp = {
            "best": best,
            "active_gate": bool(result.doa.active_gate),
            "az_grid_deg": list(map(float, result.doa.az_grid_deg)) if result.doa.az_grid_deg is not None else [],
            "el_grid_deg": list(map(float, result.doa.el_grid_deg)) if result.doa.el_grid_deg is not None else [],
        }
        if result.doa.power_map is not None:
            # el-major 2D list for compatibility
            srp["power_map"] = result.doa.power_map.tolist()
        # Determine UTC per-frame: align to recording in wall-clock time if anchors exist
        if (self._utc_anchor is not None) and (self._wall_anchor_perf is not None):
            elapsed = time.perf_counter() - self._wall_anchor_perf
            dt = self._utc_anchor + timedelta(seconds=elapsed)
            try:
                utc_iso = dt.isoformat(timespec='milliseconds')
            except TypeError:
                utc_iso = dt.isoformat()
        else:
            dt = datetime.now(timezone.utc)
            try:
                utc_iso = dt.isoformat(timespec='milliseconds')
            except TypeError:
                utc_iso = dt.isoformat()

        payload = {
            "frame_index": int(result.frame_idx),
            "levels_first5": list(map(float, result.levels_first5)) if result.levels_first5 is not None else [],
            "spec_power_mic0": list(map(float, result.spec_power_mic0)) if result.spec_power_mic0 is not None else [],
            "srp": srp,
            "total_frames": int(self._total_frames) if self._total_frames is not None else None,
            "utc_iso": utc_iso,
            # Provide sample rate redundantly to help receivers before /init arrives
            "sample_rate_hz": int(SAMPLE_RATE),
            "frame_size": int(FRAME_SIZE),
        }
        # Attach classification if present
        try:
            if result.classification is not None:
                payload["classification"] = {
                    "label": str(result.classification.label),
                    "prob": float(result.classification.prob),
                    "logits": list(map(float, result.classification.logits)) if result.classification.logits is not None else None,
                }
        except Exception:
            pass

        if self.mode == "print":
            az_len = len(srp.get("az_grid_deg", []))
            el_len = len(srp.get("el_grid_deg", []))
            pm_min = pm_max = 0.0
            if result.doa.power_map is not None:
                pm_min = float(result.doa.power_map.min())
                pm_max = float(result.doa.power_map.max())
            spec_len = len(payload["spec_power_mic0"]) if payload["spec_power_mic0"] else 0
            spec_vmax = max(payload["spec_power_mic0"]) if payload["spec_power_mic0"] else 0.0
            levels = payload["levels_first5"]
            # Disabled per-frame console prints in print mode.
            # If needed, re-enable by uncommenting the block below.
            # print(
            #     f"[FRAME {result.frame_idx}] RMS={[f'{x:.4f}' for x in levels]} | spec_len={spec_len} vmax={spec_vmax:.3e} | "
            #     f"srp_best=({best['azimuth_deg']},{best['elevation_deg']}) p={best['power']:.3g} | "
            #     f"srp_map={az_len}x{el_len} min={pm_min:.3e} max={pm_max:.3e} active={srp.get('active_gate')}"
            # )
        else:
            self._ensure_worker()
            # Non-blocking enqueue with drop-oldest policy when congested
            try:
                self._tx_queue.put_nowait({"type": "frame", "payload": payload})
            except queue.Full:
                self._drops += 1
                try:
                    # Drop the oldest frame to keep latency bounded
                    _ = self._tx_queue.get_nowait()
                    self._tx_queue.put_nowait({"type": "frame", "payload": payload})
                except Exception:
                    # Still full; log occasionally
                    now = time.perf_counter()
                    if (now - self._last_warn_ts) > 5.0:
                        self._last_warn_ts = now
                        print(f"[emitter][warn] queue congested; dropped frames total={self._drops}")
