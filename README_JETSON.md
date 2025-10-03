# NVIDIA Jetson Containerization Guide

This repository includes a containerized setup for running the real-time SRP-PHAT processing and optional classifier on NVIDIA Jetson devices (ARM64). It targets JetPack/L4T r35.x.

## Images and Base
- Base image: `nvcr.io/nvidia/l4t-ml:<L4T_TAG>` (default `r35.4.1-py3`).
- Includes CUDA/cuDNN, PyTorch, and TensorFlow builds for Jetson.
- The Dockerfile installs ALSA, PortAudio, libsndfile, ffmpeg and creates a Conda (micromamba) env for Python deps.
- `tensorflow` is intentionally filtered out from `requirements.txt` (the image already provides TF). If you don’t need TF at all, it’s optional and code handles its absence.

## Files added
- `Dockerfile`: Jetson-aware image with audio deps.
- `docker-compose.jetson.yml`: Compose service with host networking, ALSA device, RT limits, and mounts.
- `scripts/run_jetson.sh`: Convenience wrapper for build/run/logs.
- `.dockerignore`: Keeps large datasets out of the build context.

## Prerequisites
- Jetson with JetPack r35.x and NVIDIA Container Runtime (typically installed with JetPack). Verify with:
  - `docker --version`
  - `sudo docker info | grep -i runtime` shows `nvidia` runtime
- Audio interface connected and recognized on host:
  - `arecord -l`
  - `aplay -l`
- Optional: Custom ALSA config in `~/.asoundrc` for your multi-channel interface.

## Build
```bash
# From repo root
./scripts/run_jetson.sh build
```
Environment overrides:
- `L4T_TAG=r35.3.1-py3 ./scripts/run_jetson.sh build`

## Run (daemonized)
```bash
./scripts/run_jetson.sh up
```
This starts the container with:
- Host networking and IPC
- `/dev/snd` device mapped and `audio` group added
- RT ulimits (rtprio 99, memlock unlimited)
- `SYS_NICE` capability for low-latency scheduling
- Bind mounts:
  - `./field_recordings` -> `/data:ro`
  - `./catalog` and `./src` -> read-only app code
  - `./src/droneprint/models` (or `$MODELS_DIR`) -> `/models:ro`

To attach a shell:
```bash
./scripts/run_jetson.sh exec
```

## GPU access
Jetson typically defaults to the `nvidia` runtime. The compose file adds `gpus: all`. To verify inside the container:
```bash
nvidia-smi || true  # Not present on Jetson, use tegrastats
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## Audio device checks inside container
```bash
# ALSA devices
arecord -l
aplay -l
# Python sounddevice view
python -c "import sounddevice as sd; print(sd.query_devices())"
```
If devices are not visible, ensure `/dev/snd` is present on the host and mapped, and that your user is in the `audio` group on the host.

## Typical workflows

1) Webhook receiver UI
```bash
# Inside container
uvicorn webhook_receiver:app --host 0.0.0.0 --port 8000
```

2) Offline viewer (srp_gcc_viewer) on example recording
```bash
# Inside container
# Optional (diagnostic): list the file
ls -lh /app/field_recordings/20250918/250918-T002.WAV

# Run the viewer on 20250918- T013 (hex6_ring, channels 0..5)
python3 srp_gcc_viewer.py \
  /app/field_recordings/20250918/250918-T013.WAV \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15
```
Notes:
- If you see a GUI backend crash on Jetson (Tk), run headless by unsetting DISPLAY for this command:
  - `DISPLAY= python3 srp_gcc_viewer.py /app/field_recordings/20250918/250918-T013.WAV --array-ref hex6_ring --channels 0,1,2,3,4,5 --fps 15`

3) Real-time pipeline (file source) — optional webhook emission
```bash
# Inside container
python3 rt_main.py \
  --source file \
  --audio-file /app/field_recordings/20250918/250918-T013.WAV \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15
# To emit to receiver as well (recommended):
#   append --ws ws://127.0.0.1:8000/ws_ingest
# Or via HTTP webhook (alternative):
#   append --webhook http://127.0.0.1:8000
```
Notes:
- `--array-ref` must match an array id in `catalog/arrays/array_defs.json` (e.g., `hex6_ring`).
- At least 5 channels are required for SRP.

## Real-time tuning
- The compose file sets `rtprio: 99`, `memlock: -1`, and adds `SYS_NICE`. These help with audio scheduling.
- For PortAudio/ALSA buffer tuning, set environment variables as needed and/or provide `.asoundrc` with desired period/buffer sizes. The code targets frame-based processing at `FRAME_SIZE` from `src/config.py`.
- You may map a custom `~/.asoundrc` by uncommenting the bind mount in `docker-compose.jetson.yml`.

## Models
- Default `MODELS_ROOT=/models`. Mount your models directory via `MODELS_DIR=/path/to/models` when running `scripts/run_jetson.sh up`.
- `src/pipeline/drone_classifier.py` uses TensorFlow only if available; otherwise classifier is disabled gracefully.

## Stopping
```bash
./scripts/run_jetson.sh down
```

## Troubleshooting
- If `sounddevice` cannot open devices, ensure the correct device index/name is selected, and that the interface supports the sample rate specified by `src/config.py` (`SAMPLE_RATE`).
- If imports for SRP modules fail, confirm the project files are mounted read-only as in compose and `src/` is visible.
- For large build contexts, ensure `.dockerignore` is active to avoid copying `field_recordings/` into the image.

## Security note
- The service runs as root in the container to simplify audio/RT capabilities. For production hardening, consider creating a non-root user inside the image and granting only the required capabilities and group memberships.
