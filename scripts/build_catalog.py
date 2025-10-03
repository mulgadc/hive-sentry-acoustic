#!/usr/bin/env python3
"""
Build a simple document-first catalog of:
- Recording documents from WAV files (using BWF bext for UTC start)
- Telemetry documents from CSV files (start/end from time_iso; bbox from lat/lon)
- Compute overlaps when both sides have valid UTC ranges

Outputs JSON docs under catalog/:
- catalog/recordings/<id>.json
- catalog/telemetry/<id>.json

IDs are stable SHA1 hashes of absolute file paths.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
import soundfile as sf
import csv

# Local modules
# Expect to run from repo root: add src/ to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(SRC_DIR))
from wav_bwf import get_wav_start_utc  # type: ignore

CATALOG_DIR = REPO_ROOT / "catalog"
REC_DIR = CATALOG_DIR / "recordings"
TEL_DIR = CATALOG_DIR / "telemetry"

# Defaults per user context
DEFAULT_ARRAY = {
    "lat": -26.696758333,
    "lon": 152.885445,
    "alt_m": 0.0,
    "heading_deg": 0.0,
}
DEFAULT_LOCAL_OFFSET = "+10:00"  # Brisbane (no DST)


def sha1_of_path(p: Path) -> str:
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()


def parse_local_offset(offset: str) -> timezone:
    s = offset.strip()
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
    return timezone(sign * timedelta(hours=oh, minutes=om))


def wav_core_info(path: Path) -> Tuple[int, int, float]:
    """Return (channels, sample_rate, duration_s) without loading audio.
    Uses soundfile to support PCM and IEEE float WAVs.
    """
    info = sf.info(str(path))
    channels = int(info.channels)
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    duration_s = frames / float(sample_rate) if sample_rate > 0 else 0.0
    return channels, sample_rate, duration_s


def date_key_from_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return "00000000"
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")


def build_recording_doc(wav_path: Path, array: Dict[str, Any], local_offset: str) -> Dict[str, Any]:
    rid = sha1_of_path(wav_path)
    channels, sample_rate, duration_s = wav_core_info(wav_path)
    start_utc = get_wav_start_utc(str(wav_path), local_offset)
    end_utc = start_utc + timedelta(seconds=duration_s) if start_utc else None
    stat = wav_path.stat()
    doc = {
        "id": rid,
        "path": str(wav_path),
        "start_utc": start_utc.astimezone(timezone.utc).isoformat() if start_utc else None,
        "end_utc": end_utc.astimezone(timezone.utc).isoformat() if end_utc else None,
        "duration_s": round(duration_s, 3),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "array": dict(array),
        "device": None,
        "size_bytes": stat.st_size,
        "sha1": sha1_of_path(wav_path),
        "date_key": date_key_from_dt(start_utc),
        "overlaps": [],
    }
    return doc


def _parse_time_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def build_telemetry_doc_from_csv(csv_path: Path, drone_id: Optional[str] = None) -> Dict[str, Any]:
    tid = sha1_of_path(csv_path)
    row_count = 0
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    lat_min = float('inf'); lat_max = float('-inf')
    lon_min = float('inf'); lon_max = float('-inf')

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            row_count += 1
            dt = _parse_time_iso(rec.get('time_iso'))
            if dt:
                if start_dt is None or dt < start_dt:
                    start_dt = dt
                if end_dt is None or dt > end_dt:
                    end_dt = dt
            try:
                lat = float(rec.get('latitude')) if rec.get('latitude') else None
                lon = float(rec.get('longitude')) if rec.get('longitude') else None
                if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                    lat_min = min(lat_min, lat)
                    lat_max = max(lat_max, lat)
                    lon_min = min(lon_min, lon)
                    lon_max = max(lon_max, lon)
            except Exception:
                pass

    if lat_min == float('inf'):
        lat_min = lat_max = lon_min = lon_max = 0.0

    stat = csv_path.stat()
    doc = {
        "id": tid,
        "path": str(csv_path),
        "drone_id": drone_id,
        "start_utc": start_dt.isoformat() if start_dt else None,
        "end_utc": end_dt.isoformat() if end_dt else None,
        "row_count": row_count,
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "size_bytes": stat.st_size,
        "sha1": sha1_of_path(csv_path),
        "date_key": date_key_from_dt(start_dt or end_dt),
        "overlaps": [],
    }
    return doc


def compute_overlap(a_start: Optional[datetime], a_end: Optional[datetime], b_start: Optional[datetime], b_end: Optional[datetime]) -> Tuple[float, Optional[datetime], Optional[datetime]]:
    if not (a_start and a_end and b_start and b_end):
        return 0.0, None, None
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end < start:
        return 0.0, start, end
    return (end - start).total_seconds(), start, end


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build catalog docs for recordings and telemetry (CSV-only)")
    ap.add_argument("--audio-root", default=str(REPO_ROOT / "field_recordings"), help="Root directory to scan for WAVs")
    ap.add_argument("--telemetry-root", default=str(REPO_ROOT / "field_recordings"), help="Root directory to scan for telemetry CSVs")
    ap.add_argument("--local-offset", default=DEFAULT_LOCAL_OFFSET, help="Local UTC offset for WAV BWF origination time (e.g., +10:00)")
    ap.add_argument("--array-lat", type=float, default=DEFAULT_ARRAY["lat"]) 
    ap.add_argument("--array-lon", type=float, default=DEFAULT_ARRAY["lon"]) 
    ap.add_argument("--array-alt", type=float, default=DEFAULT_ARRAY["alt_m"]) 
    ap.add_argument("--array-heading", type=float, default=DEFAULT_ARRAY["heading_deg"]) 
    ap.add_argument("--array-ref", type=str, default=None, help="Array registry ID to reference (writes array.ref in recording docs)")
    ap.add_argument("--channel-map", type=str, default=None, help="Comma-separated WAV channel indices to map to the array (writes array.channel_map)")
    ap.add_argument("--dates", type=str, default=None, help="Comma-separated YYYYMMDD subdirectories to include (filter scans to these days)")
    args = ap.parse_args()

    array = {
        "lat": args.array_lat,
        "lon": args.array_lon,
        "alt_m": args.array_alt,
        "heading_deg": args.array_heading,
    }
    if args.array_ref:
        array["ref"] = args.array_ref
    if args.channel_map:
        try:
            array["channel_map"] = [int(x) for x in args.channel_map.split(',') if x.strip() != ""]
        except Exception:
            print("[WARN] Could not parse --channel-map; expected comma-separated integers (e.g., 0,1,2,3,4,5)")

    # Scan WAVs
    audio_root = Path(args.audio_root)
    wavs: List[Path] = []
    if args.dates:
        days = [d.strip() for d in args.dates.split(',') if d.strip()]
        for d in days:
            day_dir = audio_root / d
            if day_dir.exists():
                wavs.extend([p for p in day_dir.rglob("*.WAV")])
            else:
                print(f"[WARN] Day directory does not exist under audio-root: {day_dir}")
    else:
        wavs = [p for p in audio_root.rglob("*.WAV")]
    print(f"Found {len(wavs)} WAV files under {audio_root}")

    rec_docs: List[Dict[str, Any]] = []
    for wp in sorted(wavs):
        try:
            doc = build_recording_doc(wp, array, args.local_offset)
            rec_docs.append(doc)
            write_json(REC_DIR / f"{doc['id']}.json", doc)
        except Exception as e:
            print(f"[WARN] Failed WAV {wp}: {e}")

    # Scan telemetry CSVs (ignore GeoJSON for timing)
    tel_root = Path(args.telemetry_root)
    csv_paths: List[Path] = []
    if args.dates:
        days = [d.strip() for d in args.dates.split(',') if d.strip()]
        for d in days:
            day_dir = tel_root / d
            if day_dir.exists():
                csv_paths.extend([p for p in day_dir.rglob("*.csv")])
            else:
                print(f"[WARN] Day directory does not exist under telemetry-root: {day_dir}")
    else:
        csv_paths = [p for p in tel_root.rglob("*.csv")]
    print(f"Found {len(csv_paths)} telemetry CSV files under {tel_root}")

    tel_docs: List[Dict[str, Any]] = []
    for cp in sorted(csv_paths):
        try:
            doc = build_telemetry_doc_from_csv(cp)
            tel_docs.append(doc)
            write_json(TEL_DIR / f"{doc['id']}.json", doc)
        except Exception as e:
            print(f"[WARN] Failed CSV {cp}: {e}")

    # Compute overlaps (only where both sides have UTC ranges)
    # Load dt objects for comparison
    def parse_ts(s: Optional[str]) -> Optional[datetime]:
        if not s: return None
        try:
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    for r in rec_docs:
        r_start = parse_ts(r.get("start_utc"))
        r_end = parse_ts(r.get("end_utc"))
        r_dur = float(r.get("duration_s") or 0.0)
        r["overlaps"] = []  # reset
        for t in tel_docs:
            t_start = parse_ts(t.get("start_utc"))
            t_end = parse_ts(t.get("end_utc"))
            ov_s, ov_start, ov_end = compute_overlap(r_start, r_end, t_start, t_end)
            if ov_s > 0.0:
                tele_span = (t_end - t_start).total_seconds() if (t_start and t_end) else None
                t_frac = (ov_s / tele_span) if tele_span and tele_span > 0 else None
                r["overlaps"].append({
                    "telemetry_id": t["id"],
                    "telemetry_path": t.get("path"),
                    "telemetry_file": Path(t.get("path")).name if t.get("path") else None,
                    "overlap_start_utc": ov_start.isoformat() if ov_start else None,
                    "overlap_end_utc": ov_end.isoformat() if ov_end else None,
                    "overlap_s": round(ov_s, 3),
                    "telemetry_fraction": round(t_frac, 3) if t_frac is not None else None,
                })
        # rewrite updated recording doc
        write_json(REC_DIR / f"{r['id']}.json", r)

    # Also write overlaps into telemetry docs (denormalized)
    for t in tel_docs:
        t_start = parse_ts(t.get("start_utc"))
        t_end = parse_ts(t.get("end_utc"))
        t["overlaps"] = []
        for r in rec_docs:
            r_start = parse_ts(r.get("start_utc"))
            r_end = parse_ts(r.get("end_utc"))
            ov_s, ov_start, ov_end = compute_overlap(r_start, r_end, t_start, t_end)
            if ov_s > 0.0:
                r_dur = float(r.get("duration_s") or 0.0)
                a_frac = (ov_s / r_dur) if r_dur > 0 else None
                t["overlaps"].append({
                    "recording_id": r["id"],
                    "recording_path": r.get("path"),
                    "recording_file": Path(r.get("path")).name if r.get("path") else None,
                    "overlap_start_utc": ov_start.isoformat() if ov_start else None,
                    "overlap_end_utc": ov_end.isoformat() if ov_end else None,
                    "overlap_s": round(ov_s, 3),
                    "audio_fraction": round(a_frac, 3) if a_frac is not None else None,
                })
        write_json(TEL_DIR / f"{t['id']}.json", t)

    print(f"Catalog written under: {CATALOG_DIR}")


if __name__ == "__main__":
    sys.exit(main())
