"""
nd_backend: runtime-selectable array/FFT backend.
- CPU path: NumPy + SciPy
- GPU path: CuPy + CuPyX SciPy (requires CUDA runtime)

Selection:
- USE_CUDA env var ('1' enables GPU if available; default: '1' in containers typically)
- If CuPy import fails, falls back to NumPy automatically.

Provides:
- xp: numpy or cupy
- fft: numpy.fft or cupy.fft
- asarray, asnumpy
- hann_window(n, sym=False)
- take_along_axis, clip (delegated to xp)
"""
from __future__ import annotations

import os

# Default to try GPU unless explicitly disabled
_USE_CUDA = os.environ.get("USE_CUDA", "1") not in ("0", "false", "False")

xp = None  # type: ignore
fft = None  # type: ignore

_asnumpy = None  # type: ignore

# Window helper
def _hann_numpy(n: int, sym: bool = False):
    import numpy as _np
    from scipy.signal.windows import hann as _hann
    return _hann(n, sym=sym).astype(_np.float32, copy=False)


def _hann_cupy(n: int, sym: bool = False):
    import cupy as _cp
    from cupyx.scipy.signal.windows import hann as _hann
    # cupyx hann returns cupy.ndarray; cast to float32
    return _hann(n, sym=sym).astype(_cp.float32, copy=False)


try:
    if _USE_CUDA:
        import cupy as _cp  # type: ignore
        import cupy.fft as _cfft  # type: ignore
        xp = _cp
        fft = _cfft
        _asnumpy = _cp.asnumpy
        hann_window = _hann_cupy
        try:
            _dev = _cp.cuda.runtime.getDevice()
            _props = _cp.cuda.runtime.getDeviceProperties(_dev)
            _name = _props.get('name', b'CUDA').decode() if isinstance(_props.get('name'), (bytes, bytearray)) else str(_props.get('name'))
            print(f"[nd_backend] Using CuPy GPU backend on device: {_name}")
        except Exception:
            print("[nd_backend] Using CuPy GPU backend")
    else:
        raise ImportError("USE_CUDA disabled")
except Exception:
    import numpy as _np  # type: ignore
    import numpy.fft as _nfft  # type: ignore
    xp = _np
    fft = _nfft
    _asnumpy = lambda a: a  # no-op for numpy
    hann_window = _hann_numpy
    print("[nd_backend] Using NumPy CPU backend")


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype)


def asnumpy(a):
    return _asnumpy(a)


def take_along_axis(arr, indices, axis):
    return xp.take_along_axis(arr, indices, axis=axis)


def clip(a, a_min, a_max, out=None):
    return xp.clip(a, a_min, a_max, out=out)
