# Jetson Setup and First-Run (via GitHub)

This guide documents how to run the project on a Jetson using the Docker setup in this repo. USB copying is no longer required because we now pull code directly from GitHub.

Note: The older USB transfer steps are kept at the bottom for reference but are deprecated.

## Files and folders to copy onto the USB
Copy the repository directory with the following items at minimum:

- Docker and orchestration
  - Dockerfile
  - docker-compose.jetson.yml
  - .dockerignore
  - scripts/run_jetson.sh
- Python code and config
  - src/
  - config.py
  - requirements.txt
  - rt_main.py (if you plan to use it)
  - srp_gcc_viewer.py
  - webhook_receiver.py
  - webhook_emitter.py
- Catalog and any sample data you want available
  - catalog/
  - field_recordings/ (optional; you can also leave this out and mount later)
- Models (choose one of these two approaches)
  - Option A (simple): Copy your models into src/droneprint/models/ so it travels with the repo.
  - Option B (separate): Copy your models folder alongside the repo (e.g., /home/jetson/models). You’ll point docker-compose to it via MODELS_DIR later.

Your USB will have, for example:

- /USB/mulga-rtp/
  - Dockerfile
  - docker-compose.jetson.yml
  - .dockerignore
  - scripts/run_jetson.sh
  - requirements.txt
  - config.py
  - srp_gcc_viewer.py
  - webhook_receiver.py
  - webhook_emitter.py
  - src/
  - catalog/
  - field_recordings/ (optional)
  - src/droneprint/models/ (or you’ll bring a separate models folder)

## Prepare the Jetson (one-time)
- Ensure Docker + NVIDIA Container Runtime are installed (usually part of JetPack).
  - Check:
    - `docker --version`
    - `sudo docker info | grep -i runtime`  # should mention nvidia
  - Ensure docker compose plugin is available:
    - `docker compose version`
- Plug in your audio interface and verify on the host:
  - `arecord -l`
  - `aplay -l`

## Quick start (GitHub)
- Clone the repository on the Jetson and start the container:
```bash
git clone https://github.com/mulgadc/srp-phat-workbench.git
cd srp-phat-workbench
# Optional: point MODELS_DIR to a models folder on the host, otherwise defaults to repo models
MODELS_DIR=./src/droneprint/models ./scripts/run_jetson.sh up
# Attach a shell inside the running container
./scripts/run_jetson.sh exec
```

## Build the container on the Jetson
- From the project root:
```bash
cd ~/mulga-rtp
# Default JetPack tag (r35.4.1-py3). Adjust via env if needed.
./scripts/run_jetson.sh build
# Alternate (if your Jetson has a different L4T tag):
L4T_TAG=r35.3.1-py3 ./scripts/run_jetson.sh build
```

## Start the container
- Start with host networking, ALSA access, and mounts:
```bash
# If models are in a separate folder (Option B), set MODELS_DIR:
MODELS_DIR=~/models ./scripts/run_jetson.sh up
# If models live inside the repo (Option A), just:
./scripts/run_jetson.sh up
```
- Attach a shell inside the running container:
```bash
./scripts/run_jetson.sh exec
```

## First-time verification inside the container
- Check ALSA devices:
```bash
arecord -l
aplay -l
```
- Check Python can see audio devices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## First run A: Start the webhook plotting receiver
- In one terminal inside the container:
```bash
python webhook_receiver.py --host 0.0.0.0 --port 8000
```
- In another terminal inside the container, you can either run the offline viewer or the real-time pipeline to emit to the receiver.

### Option A1: Offline viewer (srp_gcc_viewer, no webhook)
```bash
python srp_gcc_viewer.py \
  /app/field_recordings/20250918/250918-T013.WAV \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15
```

### Option A2: Real-time pipeline (rt_main) emitting to the receiver
```bash
python rt_main.py \
  --source file \
  --audio-file /app/field_recordings/20250918/250918-T013.WAV \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15 \
  --webhook http://127.0.0.1:8000
```
Notes:
- WAV files are available under `/app/field_recordings` inside the container (bind-mounted from the repo).
- Ensure the `--array-ref` exists in `catalog/arrays/array_defs.json` (e.g., `hex6_ring`).
- At least 5 channels are required for SRP.

## First run B: Offline viewer only (headless-safe)
- Run the viewer against the example WAV (uses Agg backend by default when no DISPLAY):
```bash
python srp_gcc_viewer.py \
  /app/field_recordings/20250918/250918-T013.WAV \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15
```

## First run option C: Real-time capture via ALSA input
- The viewer supports STREAM mode with `sounddevice`. Typical flow:
  - List audio devices (viewer supports listing; check README or run interactively).
  - Provide the correct input device name/index and the array reference.
- Example (replace <device-name-or-index> and <array-id>):
```bash
python srp_gcc_viewer.py \
  --array-ref hex6 \
  --fps 15 \
  --list-devices \
  --source-mode STREAM \
  --input-device "<device-name-or-index>"
```
Tip for low latency:
- If you have a custom `~/.asoundrc` on the host, you can map it into the container. In `docker-compose.jetson.yml`, uncomment:
- `${HOME}/.asoundrc:/root/.asoundrc:ro`

## Where things live at runtime (inside the container)
- Code: `/app/`
- Catalog: `/app/catalog`
- Field recordings: `/app/field_recordings` (bind-mount of repo `./field_recordings`)
- Models: `/models` (bind-mount of host `${MODELS_DIR}` or repo default)
- `MODELS_ROOT` env defaults to `/models` (ensure it points to the folder that contains the `.h5` files, e.g., models_smoke).

---

## (Deprecated) Copy from USB to the Jetson
- Prefer cloning from GitHub now. If you still need to use USB, the following legacy steps apply.
- On the Jetson, mount the USB (it usually auto-mounts under `/media/jetson/...`). Then copy:
```bash
# Example paths — adjust to your actual mount point and username
cp -r "/media/jetson/YOUR_USB_LABEL/mulga-rtp" ~/mulga-rtp
# Optional: copy external models directory if using Option B
cp -r "/media/jetson/YOUR_USB_LABEL/models" ~/models
```

## If you didn’t copy field recordings or want to use a different path
- You can point the compose file’s `/data` mapping to a different host folder:
  - Edit `docker-compose.jetson.yml`:
    - volumes:
      - `/path/to/your/recordings:/data:ro`

## If you didn’t copy models to `src/droneprint/models`
- Use the `MODELS_DIR` override when starting:
```bash
MODELS_DIR=/home/jetson/models ./scripts/run_jetson.sh up
```

## Troubleshooting quick checks
- Audio not visible in container:
  - Ensure `/dev/snd` is present on host and docker-compose has `devices: - /dev/snd`
  - Make sure your user belongs to `audio` group on host (or run compose under sudo).
- TensorFlow not needed:
  - The code gracefully disables TF-based classifier if TF isn’t importable. The image avoids installing TF via pip and relies on Jetson’s `l4t-ml` base.
- Real-time stutters:
  - Confirm `rtprio` and `memlock` ulimits are applied (compose).
  - Consider mapping a tuned `~/.asoundrc` for your device with smaller period/buffer sizes.

That’s it. If you share your JetPack/L4T version and audio interface model, I can pre-set the `L4T_TAG` in the compose and draft a minimal `.asoundrc` tuned for sub-10 ms latency.
