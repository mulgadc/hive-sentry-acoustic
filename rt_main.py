from __future__ import annotations

import argparse

from src.config import SRP_AZ_RANGE, SRP_EL_RANGE
from src.catalog_utils import resolve_array_positions_from_catalog
from src.pipeline.audio_source import AudioSource
from src.pipeline.drone_detector import DroneDetector
from src.pipeline.doa_engine import DOAEngine
from src.pipeline.drone_classifier import DroneClassifier
from src.pipeline.emitter import Emitter
from src.pipeline.runner import PipelineRunner


def main() -> int:
    p = argparse.ArgumentParser(description="Real-time SRP-PHAT pipeline runner")
    p.add_argument("--source", choices=["file", "stream"], default="file")
    p.add_argument("--audio-file", type=str, help="Path to WAV file (for file source)")
    p.add_argument("--array-ref", type=str, required=True, help="Array ID from catalog/arrays/array_defs.json")
    p.add_argument("--channels", type=str, default="", help="Comma-separated channel indices (default: all from geometry)")
    p.add_argument("--webhook", type=str, default="", help="Webhook base URL (omit for print mode)")
    p.add_argument("--ws", type=str, default="", help="WebSocket URL (e.g., ws://127.0.0.1:8000 or ws://host:port/ws_ingest). If provided, emitter uses WebSocket instead of HTTP.")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--post-timeout", type=float, default=0.5, help="Webhook POST timeout in seconds (non-blocking)")
    p.add_argument("--device", type=str, default=None, help="Input device index or name for stream source (auto-select if omitted)")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--audio-local-utc-offset", type=str, default="+10:00", help="Local UTC offset used to parse WAV BWF OriginationTime for UTC alignment (e.g., +10:00)")
    p.add_argument("--no-classifier", action="store_true", help="Disable classifier (for performance comparison)")
    args = p.parse_args()

    if args.list_devices:
        try:
            devices = AudioSource.list_devices()
            print("Available audio devices:")
            for i, d in enumerate(devices):
                name = d.get('name', '')
                print(f"  {i}: {name} (in: {d.get('max_input_channels', 0)}, out: {d.get('max_output_channels', 0)})")
        except Exception as e:
            print(f"Could not query audio devices: {e}")
        return 0

    if args.source == "file" and not args.audio_file:
        p.error("--audio-file is required for file source")

    # Geometry
    positions = resolve_array_positions_from_catalog(args.array_ref)
    num_mics = positions.shape[0]

    # Channels
    if args.channels.strip():
        mic_channels = list(map(int, args.channels.split(',')))
    else:
        mic_channels = list(range(num_mics))

    # Source
    if args.source == "file":
        source = AudioSource(backend="file", file_path=args.audio_file, channels=mic_channels)
    else:
        # STREAM: device can be index or name; AudioSource will auto-select if None
        device = None
        if args.device is not None and len(str(args.device).strip()) > 0:
            # allow numeric index or name string
            try:
                device = int(args.device)
            except Exception:
                device = str(args.device)
        source = AudioSource(backend="stream", channels=mic_channels, device=device)

    # Detector
    detector = DroneDetector()

    # DOA engine
    doa = DOAEngine(
        array_positions=positions,
        az_range=tuple(SRP_AZ_RANGE),
        el_range=tuple(SRP_EL_RANGE),
        az_step=1.0,
        el_step=1.0,
    )

    # Classifier (toggleable)
    classifier = DroneClassifier(enabled=(not args.no_classifier))

    # Emitter
    source_mode_str = "WAV" if args.source == "file" else "STREAM"
    if args.webhook or args.ws:
        emitter = Emitter(mode="webhook", webhook_url=(args.webhook or None), post_timeout=float(args.post_timeout), source_mode=source_mode_str, ws_url=(args.ws or None))
    else:
        emitter = Emitter(mode="print", source_mode=source_mode_str)

    # If replaying from file, try to align UTC to the recording start using BWF metadata
    if args.source == "file" and args.audio_file:
        try:
            from src.wav_bwf import get_wav_start_utc
            start_utc = get_wav_start_utc(args.audio_file, args.audio_local_utc_offset)
        except Exception:
            start_utc = None
        if hasattr(emitter, "set_time_base"):
            emitter.set_time_base(start_utc)

    # Runner
    runner = PipelineRunner(
        source=source,
        detector=detector,
        doa=doa,
        classifier=classifier,
        emitter=emitter,
        target_fps=float(args.fps),
        total_frames=None,
    )

    runner.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
