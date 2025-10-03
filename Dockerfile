# NVIDIA Jetson (ARM64) container for real-time SRP-PHAT processing
# Base includes PyTorch and TensorFlow builds for Jetson (JetPack r36.x)
ARG L4T_TAG=r36.2.0-py3
FROM nvcr.io/nvidia/l4t-ml:${L4T_TAG}

# Prevent tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps for audio, performance, and build tools
# - ALSA and PortAudio for low-latency capture/playback
# - libsndfile for soundfile
# - ffmpeg for librosa I/O
# - libatlas-base-dev for optimized numpy on Jetson
# - git, curl for tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 libasound2-dev alsa-utils \
    libportaudio2 portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    libatlas-base-dev \
    python3-tk tk \
    bzip2 \
    build-essential \
    pkg-config \
    python3-dev \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Optional micromamba usage (Conda on ARM64). Default OFF to minimize surprises.
#   Enable with: docker build --build-arg USE_MAMBA=1 ...
ARG USE_MAMBA=0
ENV USE_MAMBA=${USE_MAMBA} \
    MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_DOCKERFILE_ACTIVATE=1

RUN set -eux; \
    echo "USE_MAMBA=${USE_MAMBA}"; \
    if [ "${USE_MAMBA}" = "1" ]; then \
      echo "MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}"; \
      # Use mambaforge installer for ARM64 - more reliable than micromamba direct download
      curl --fail --location --show-error --retry 3 -o /tmp/mambaforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh"; \
      bash /tmp/mambaforge.sh -b -p ${MAMBA_ROOT_PREFIX}; \
      rm /tmp/mambaforge.sh; \
      ${MAMBA_ROOT_PREFIX}/bin/mamba --version; \
      ${MAMBA_ROOT_PREFIX}/bin/mamba shell init -s bash; \
    else \
      echo "Skipping micromamba installation (USE_MAMBA=0)."; \
    fi

## Create env with conda-forge; prefer Conda for core scientific stack on ARM64
# Note: PyTorch is already provided in base image; we keep it in base env.
RUN if [ "${USE_MAMBA}" = "1" ]; then \
      ${MAMBA_ROOT_PREFIX}/bin/mamba create -y -n rtp -c conda-forge \
        python=3.10 \
        numpy scipy matplotlib \
        librosa \
        fastapi uvicorn \
        plotly \
        websocket-client \
        soundfile \
        python-sounddevice \
        pip; \
    else \
      echo "Using base Python with pip-only (no mamba env)."; \
    fi && true

SHELL ["/bin/bash", "-lc"]

# Activate env and install any pip-only extras (none required beyond requirements.txt)
WORKDIR /app
COPY requirements.txt ./
RUN if [ "${USE_MAMBA}" = "1" ]; then \
      source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh; \
      conda activate rtp; \
      PYBIN=python; \
    else \
      PYBIN=python3; \
    fi; \
    awk 'BEGIN{print "Filtered requirements (no tensorflow)"} !/^tensorflow/{print > "requirements.jetson.txt"} END{}' requirements.txt >/dev/null 2>&1 || true; \
    "$PYBIN" -m pip install --no-cache-dir -r requirements.jetson.txt; \
    # Ensure only one CuPy variant is installed. Remove any preinstalled/conflicting CuPy first, then install CUDA12 variant.
    "$PYBIN" -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x >/dev/null 2>&1 || true; \
    "$PYBIN" -m pip install --no-cache-dir cupy-cuda12x; \
    true

# Copy project
COPY . /app

# Set defaults for models root and ALSA tuning
ENV MODELS_ROOT=/app/src/droneprint/models \
    MPLBACKEND=Agg \
    USE_CUDA=1 \
    # Prefer low-latency ALSA behavior (tune as needed)
    AUDIO_LATENCY_MS=8 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Expose FastAPI port if you use webhook_receiver.py
EXPOSE 8000

# No ENTRYPOINT - let docker-compose command handle startup
