# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv srp-phat-env
source srp-phat-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python wavorstream.py
```

## Project Architecture

This is a real-time sound source localization system using Steered Response Power with Phase Transform (SRP-PHAT) algorithm. The system captures audio from microphones, processes it for direction-of-arrival (DOA) estimation, and provides real-time visualization.

### Core Components

- **wavorstream.py**: Main application with real-time audio processing, visualization, and user interface
- **src/srp_phat.py**: Core SRP-PHAT algorithm implementation for direction-of-arrival estimation
- **src/gcc_phat.py**: Generalized Cross-Correlation with Phase Transform for microphone pairs
- **src/build_array.py**: Microphone array geometry configuration (cross array positions)
- **src/build_steering_grid.py**: Azimuth/elevation search grid generation for beamforming
- **src/helpers.py**: Utility functions for coordinate transformations and delay calculations
- **src/audio.py**: Audio I/O handling (currently minimal)
- **src/processor.py**: Real-time processing components (currently minimal)

### Key Technical Details

- **Audio Processing**: 48kHz sample rate, 50ms windows (2400 samples)
- **SRP-PHAT Optimization**: Downsampled to 24kHz for DOA computation, 128-point FFT
- **Microphone Array**: Cross-pattern array with configurable azimuth offset (default 150°)
- **Search Grid**: 181 azimuth × 46 elevation points (3° resolution)
- **Visualization**: Real-time magnitude spectrogram, mel spectrogram, and SRP-PHAT heatmap

### Audio Input Modes

The system supports two audio input modes via `STREAM_SOURCE` variable:
- `"MIC"`: Live microphone input with multi-channel support
- `"WAV"`: File playback from `my_audio.WAV`

### Performance Optimizations

- SRP computation throttling (every 4th frame for 20Hz updates)
- Bounded queues with drop-oldest policy to prevent memory growth
- Memory-mapped audio buffers for ARM64 deployment
- ALSA audio backend configuration for low-latency capture

### ARM64 Deployment Focus

The codebase is optimized for ARM64 platforms (Raspberry Pi/NVIDIA Jetson):
- All dependencies have ARM64 wheels available
- ALSA audio backend for low-latency capture
- GPIO-based microphone array interface support
- Cross-compilation workflow compatibility