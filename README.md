# SRP-GCC-PHAT Viewer

Version: 4.0

This script provides a detailed visualization of Generalized Cross-Correlation with Phase Transform (GCC-PHAT) for a multi-microphone array and computes Direction-of-Arrival (DOA) estimates using Steered Response Power (SRP-PHAT). It is designed to analyze pre-recorded multi-channel audio files.

## Features

-   **Multi-Pair GCC-PHAT**: Visualizes GCC-PHAT curves for all 10 unique pairs from a 5-channel microphone array.
-   **SRP-PHAT Heatmap**: Displays a 2D heatmap of sound source power across azimuth and elevation.
-   **DOA Estimation**: Identifies and displays the most likely direction of arrival.
-   **Interactive Playback**: Full control over audio playback, including play, pause, seek, and speed control.
-   **TDOA Reference**: Shows expected Time Difference of Arrival (TDOA) lags for a target direction, allowing for system calibration and analysis.
-   **Audio Filtering**: Includes options for voice-band and spatial anti-aliasing filters.
-   **Telemetry Overlay (Auto-Select)**: Optionally overlays GPS/heading telemetry, auto-selecting the best matching CSV from the catalog.

## Prerequisites

Ensure you have Python 3 installed. You will also need to install the required Python packages.

```bash
# It is recommended to use a virtual environment
python3 -m venv srp-phat-env
source srp-phat-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the script from your terminal, providing the path to a multi-channel audio file.

```bash
python srp_gcc_viewer.py /path/to/your/audiofile.wav
```

### Webhook receiver (uvicorn)

To visualize real-time output from the pipeline, start the minimal receiver UI with uvicorn:

```bash
uvicorn webhook_receiver:app --host 127.0.0.1 --port 8000
```

Then open the dashboard at:

- http://127.0.0.1:8000

Emitter endpoints exposed by the receiver:

- WebSocket ingest (recommended): `ws://127.0.0.1:8000/ws_ingest`
- HTTP POST (alternative): `/init` and `/frame`

Example: run the real-time pipeline and emit via WebSocket

```bash
python3 rt_main.py \
  --source stream \
  --array-ref hex6_ring \
  --channels 0,1,2,3,4,5 \
  --fps 15 \
  --ws ws://127.0.0.1:8000/ws_ingest
```

### Examples

- File-based analysis (reference to a file):

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4
```

- Realtime processing (live input device):

List devices first to find the input device index:

```bash
python list_audio_devices.py
```

Or directly via the viewer:

```bash
python srp_gcc_viewer.py --source STREAM --list-devices
```

Then run the realtime viewer with `srp_gcc_viewer.py` in STREAM mode (example uses device 0 and 5 channels):

```bash
python srp_gcc_viewer.py \
  --source STREAM \
  --device 0 \
  --channels 0,1,2,3,4
```

Auto-select the input device (omit `--device`):
python srp_gcc_viewer.py \
  --source STREAM \
  --channels 0,1,2,3,4

```bash
python srp_gcc_viewer.py \
  --source STREAM \
  --channels 0,1,2,3,4
```

Note: If you omit `--device`, the viewer will auto-select an input device:

- It prefers a device named like “Zoom F8n Pro” (case-insensitive).
- Otherwise it picks the first device with ≥ 5 input channels.
- You can always override with `--device <index>`.

- Build the telemetry catalog:

```bash
python scripts/build_catalog.py \
  --audio-root field_recordings \
  --telemetry-root field_recordings \
  --local-offset +10:00 \
  --array-lat -26.696758333 --array-lon 152.885445 --array-alt 0.0 \
  --array-heading 0
```

- Run with telemetry settings:

Auto-select (default) with explicit local UTC offset for BWF parsing:

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4 \
  --audio-local-utc-offset +10:00
```

#### Arrays registry and per-recording geometry

You can define reusable microphone arrays in `catalog/arrays/array_defs.json` and reference them from recording documents. This keeps your recordings self-consistent even if you switch arrays between days.

- Registry example (`catalog/arrays/array_defs.json`):

```json
{
  "version": 1,
  "units": "meters",
  "frame": "ENU",
  "arrays": [
    {
      "id": "cross5_spike",
      "positions": [
        [ 0.00,  0.65, 0.78],
        [-0.60,  0.00, 0.78],
        [ 0.00, -0.60, 0.78],
        [ 0.65,  0.00, 0.78],
        [ 0.00,  0.00, 1.99]
      ],
      "mic_labels": ["N", "W", "S", "E", "Spike"]
    },
    {
      "id": "cross6_spike2",
      "positions": [
        [ 0.00,  0.61, 0.78],
        [-0.55,  0.00, 0.78],
        [ 0.00, -0.55, 0.78],
        [ 0.61,  0.00, 0.78],
        [ 0.00,  0.00, 1.99],
        [ 0.00,  0.00, 2.00]
      ],
      "mic_labels": ["N", "W", "S", "E", "Spike", "Spike2"]
    }
  ]
}
```

- Recording JSON can reference the registry and provide per-recording fields:

```json
"array": {
  "ref": "cross6_spike2",
  "heading_deg": 130.0,
  "channel_map": [0,1,2,3,4,5],
  "lat": -26.696758333,
  "lon": 152.885445,
  "alt_m": 0.0
}
```

- Alternatively, inline positions in the recording (one-off arrays):

```json
"array": {
  "positions": [[...],[...]],
  "heading_deg": 0.0,
  "channel_map": [0,1,2,3,4]
}
```

Viewer behavior:

- If `array.positions` exists, it is used directly (after applying `heading_deg`).
- Else if `array.ref` exists, it is resolved from `catalog/arrays/array_defs.json`.
- If `--channels` is omitted, `array.channel_map` is used; otherwise default to `0..N-1` where `N` is the number of positions.
- CLI flags still override catalog values where applicable.

Override catalog values explicitly via CLI:

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4 \
  --array-heading 45 \
  --array-lat -26.7000 --array-lon 152.8800 --array-alt 30
```

Note (STREAM mode): When running realtime (`--source STREAM`), there is no WAV path to look up in the catalog, so array settings are not auto-populated. Provide `--array-heading` (and optionally `--array-lat/--array-lon/--array-alt`) if needed.

Manually specify a telemetry CSV (overrides auto-select):

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4 \
  --telemetry field_recordings/20250905/FlightRecord_2025-09-05_[11-03-15].csv
```

Disable auto-selection entirely:

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4 \
  --no-telemetry-auto
```

#### Array orientation and coordinates (auto-populated)

When you open a WAV (not STREAM), the viewer will now auto-populate the array latitude, longitude, altitude, and heading from the matching recording JSON in `catalog/recordings/` if available. The catalog file contains an `array` block like:

```json
"array": {
  "lat": -26.696758333,
  "lon": 152.885445,
  "alt_m": 0.0,
  "heading_deg": 0.0
}
```

Precedence:

- CLI flags `--array-lat/--array-lon/--array-alt` always override catalog values.
- `--array-heading` overrides the catalog heading. If you omit it and the catalog has `heading_deg`, that value is used. If neither is present, the default heading is 0°.

Example (no explicit array flags; uses catalog array settings):

```bash
python srp_gcc_viewer.py field_recordings/20250905/250905-T002.WAV \
  --channels 0,1,2,3,4 \
  --audio-local-utc-offset +10:00
```

### Command-Line Arguments

| Argument            | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `audio_file`        | **(Required)** Path to the audio file (WAV, FLAC, MP3, etc.).               |
| `--channels`        | Comma-separated list of 5 channel indices to use. Default: `0,1,2,3,4`.     |
| `--speed`           | Playback speed multiplier. Default: `1.0`.                                  |
| `--no-audio`        | Disable audio playback (visualization only).                                |
| `--no-filter`       | Disable the default bandpass filter.                                        |
| `--playback-device` | The numerical ID of the audio device to use for playback.                   |

### Interactive Controls

Once the viewer is running, you can use the following keyboard and mouse controls:

| Key / Action              | Function                                                              |
| ------------------------- | --------------------------------------------------------------------- |
| `Spacebar`                | Play or pause the audio playback.                                     |
| `r`                       | Reset the playback to the beginning of the file.                      |
| `a`                       | Toggle audio playback on or off.                                      |
| `q`                       | Quit the application.                                                 |
| `t`                       | Toggle the visibility of the TDOA reference lines.                    |
| `←` / `→` (Arrow Keys)    | Adjust the azimuth of the TDOA reference target by +/- 5 degrees.     |
| `↑` / `↓` (Arrow Keys)    | Adjust the elevation of the TDOA reference target by +/- 5 degrees.   |
| `b`                       | Toggle the voice-band filter (300-3400 Hz) on or off.                 |
| `l`                       | Toggle the spatial anti-aliasing low-pass filter (300 Hz) on or off.  |
| `Click on Progress Bar`   | Seek to a specific frame in the audio file.                           |

## Telemetry Overlay and Catalog Auto-Selection

The viewer can overlay telemetry (e.g., GPS/heading) against the audio timeline. Telemetry is read from CSV files with UTC timestamps (`time_iso`), and can be selected automatically from a local catalog.

### Build the catalog (once per dataset)

Use the helper script to index audio and telemetry files and compute overlaps:

```bash
python scripts/build_catalog.py \
  --audio-root field_recordings \
  --telemetry-root field_recordings \
  --local-offset +10:00 \
  --array-lat -26.696758333 --array-lon 152.885445 --array-alt 0.0 \
  --array-heading 0
```

This writes JSON documents under `catalog/recordings/` and `catalog/telemetry/` that the viewer uses for auto-selection.

#### Per-day catalog builds for multiple arrays

If your field recordings contain different arrays on different days, run the catalog builder per day (or day set) with the appropriate array reference and channel map.

- Example: build for two days using the 5‑mic array:

```bash
python scripts/build_catalog.py \
  --audio-root field_recordings \
  --telemetry-root field_recordings \
  --local-offset +10:00 \
  --dates 20250904,20250905 \
  --array-ref cross5_spike \
  --channel-map 0,1,2,3,4 \
  --array-lat -26.696758333 --array-lon 152.885445 --array-alt 0.0 \
  --array-heading 130
```

- Example: build for another day using the 6‑mic array:

```bash
python scripts/build_catalog.py \
  --audio-root field_recordings \
  --telemetry-root field_recordings \
  --local-offset +10:00 \
  --dates 20251001 \
  --array-ref cross6_spike2 \
  --channel-map 0,1,2,3,4,5 \
  --array-lat -26.696758333 --array-lon 152.885445 --array-alt 0.0 \
  --array-heading 130
```

The script will only scan the specified `--dates` subdirectories and will write `array.ref`, `array.channel_map`, and array coordinates into each recording JSON.

### How telemetry selection works

- If you pass `--telemetry <path/to/file.csv>`, that file is used (auto-selection is bypassed).
- Otherwise, with auto-selection enabled (default), the viewer:
  1. Looks up the audio file in `catalog/recordings/*.json` and chooses the telemetry with the largest `overlap_s`.
  2. If no overlaps are present, it falls back to time-range matching using `catalog/telemetry/*.json`:
     - Prefer a telemetry whose `[start_utc, end_utc]` contains the audio start time.
     - If none contains it, pick the nearest by boundary time difference.

The selected path is printed, e.g.:

```
✓ Telemetry auto-selected from catalog: field_recordings/20250905/FlightRecord_2025-09-05_[11-03-15].csv
Loading telemetry from field_recordings/20250905/FlightRecord_2025-09-05_[11-03-15].csv
```

### Telemetry-related CLI flags

- `--telemetry PATH`
  - Manually specify a telemetry CSV. Overrides auto-selection.
- `--telemetry-auto` / `--no-telemetry-auto`
  - Enable/disable auto-selection from the catalog (default: enabled).
- `--audio-local-utc-offset "+10:00"`
  - Local UTC offset used for parsing BWF `origination_time` when deriving audio start times.

Notes:

- Auto-selection requires that you have built the catalog and that the recording JSON for your WAV matches the WAV path (absolute or repository-relative).
- When falling back to the nearest telemetry (no strict overlap), the overlay may clamp at the start or end; this is expected if the windows do not overlap.
