"""
Minimal BWF (Broadcast WAV) parser to determine audio start time in UTC.
Parses 'bext' chunk for OriginationDate, OriginationTime (local), and TimeReference (samples).
"""
from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


def _read_chunks(fp):
    # Assumes standard RIFF with little endian
    riff, size, wave = struct.unpack('<4sI4s', fp.read(12))
    if riff != b'RIFF' or wave != b'WAVE':
        raise ValueError('Not a WAVE RIFF file')
    # Iterate chunks
    while True:
        hdr = fp.read(8)
        if len(hdr) < 8:
            return
        cid, csize = struct.unpack('<4sI', hdr)
        data_pos = fp.tell()
        yield cid, csize, data_pos
        # Seek to next chunk (pad to even)
        fp.seek(data_pos + csize + (csize % 2))


def _parse_fmt(fp, data_pos: int, csize: int) -> Tuple[int, int]:
    fp.seek(data_pos)
    # PCM fmt at least 16 bytes
    data = fp.read(min(csize, 40))
    # format, channels, sampleRate, byteRate, blockAlign, bitsPerSample ...
    if len(data) < 16:
        raise ValueError('Invalid fmt chunk')
    fmt_tag, channels, sample_rate = struct.unpack('<HHI', data[:8])
    return sample_rate, channels


def _parse_bext(fp, data_pos: int, csize: int):
    fp.seek(data_pos)
    data = fp.read(csize)
    # bext layout (Broadcast Audio Extension):
    # https://tech.ebu.ch/docs/tech/tech3285.pdf
    # Fields: Description[256], Originator[32], OriginatorReference[32], OriginationDate[10], OriginationTime[8], TimeReference(2x uint32), ...
    if len(data) < 256+32+32+10+8+8:
        return None
    off = 256 + 32 + 32
    orig_date = data[off:off+10].decode('ascii', errors='ignore')
    off += 10
    orig_time = data[off:off+8].decode('ascii', errors='ignore')
    off += 8
    time_ref_low, time_ref_high = struct.unpack('<II', data[off:off+8])
    time_reference = (time_ref_high << 32) | time_ref_low
    return orig_date.strip(), orig_time.strip(), time_reference


def get_wav_start_utc(path: str, local_utc_offset_str: Optional[str]) -> Optional[datetime]:
    """Return UTC datetime for the timestamp of the first audio sample.
    local_utc_offset_str examples: '+10:00', '+10', '+10:30', '-07:00'. If None, returns None.
    NOTE: We intentionally use OriginationDate+OriginationTime directly and ignore TimeReference,
    because on our recorder OriginationTime already corresponds to the actual recording start.
    """
    try:
        with open(path, 'rb') as fp:
            sample_rate = None
            bext = None
            for cid, csize, data_pos in _read_chunks(fp):
                if cid == b'fmt ' and sample_rate is None:
                    sample_rate, _ = _parse_fmt(fp, data_pos, csize)
                elif cid == b'bext' and bext is None:
                    bext = _parse_bext(fp, data_pos, csize)
            if bext is None or sample_rate is None:
                return None
            orig_date, orig_time, time_reference = bext
            # OriginationDate: 'YYYY-MM-DD', OriginationTime: 'HH:MM:SS'
            try:
                dt_local = datetime.strptime(orig_date + ' ' + orig_time, '%Y-%m-%d %H:%M:%S')
            except Exception:
                return None
            if not local_utc_offset_str:
                return None
            # parse offset
            s = local_utc_offset_str.strip()
            sign = 1
            if s.startswith('-'):
                sign = -1
                s = s[1:]
            elif s.startswith('+'):
                s = s[1:]
            if ':' in s:
                h, m = s.split(':', 1)
                oh, om = int(h), int(m)
            else:
                oh, om = int(s), 0
            tz = timezone(sign * timedelta(hours=oh, minutes=om))
            dt_local = dt_local.replace(tzinfo=tz)
            # Use OriginationDate+OriginationTime as the first-sample time (ignore TimeReference).
            start_dt_local = dt_local
            start_dt_utc = start_dt_local.astimezone(timezone.utc)
            return start_dt_utc
    except Exception:
        return None
