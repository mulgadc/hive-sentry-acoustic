"""
Telemetry utilities: load drone telemetry CSV, sanitize, compute az/el relative to array,
and interpolate az/el at arbitrary UTC times.

No external dependencies. Uses WGS84 formulas for ECEF/ENU conversions.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional
import math

# WGS84 constants
_A = 6378137.0  # semi-major axis (m)
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)  # first eccentricity squared


def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO 8601 timestamp with timezone to UTC-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        # Assume already UTC if no tz
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s = math.sin(lat)
    c = math.cos(lat)
    N = _A / math.sqrt(1 - _E2 * s * s)
    x = (N + h_m) * c * math.cos(lon)
    y = (N + h_m) * c * math.sin(lon)
    z = (N * (1 - _E2) + h_m) * s
    return x, y, z


def ecef_to_enu(x: float, y: float, z: float,
                ref_lat_deg: float, ref_lon_deg: float, ref_h_m: float) -> Tuple[float, float, float]:
    xr, yr, zr = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_h_m)
    dx, dy, dz = x - xr, y - yr, z - zr
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # ENU rotation
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def enu_to_azel(e: float, n: float, u: float) -> Tuple[float, float]:
    az = math.degrees(math.atan2(e, n))
    if az < 0:
        az += 360.0
    el = math.degrees(math.atan2(u, math.hypot(e, n)))
    # Clamp elevation to [0, 90] for display, though negative is physically possible
    el = max(-90.0, min(90.0, el))
    return az, el


@dataclass
class TelemetryRow:
    t_utc: datetime
    lat: float
    lon: float
    alt_m: float


class Telemetry:
    def __init__(self, rows: List[TelemetryRow]):
        if not rows:
            raise ValueError("No telemetry rows")
        # Ensure sorted by time
        rows.sort(key=lambda r: r.t_utc)
        self.rows = rows
        self.times = [r.t_utc for r in rows]
        self.lats = [r.lat for r in rows]
        self.lons = [r.lon for r in rows]
        self.alts = [r.alt_m for r in rows]

    @staticmethod
    def load_csv(path: str) -> "Telemetry":
        rows: List[TelemetryRow] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                try:
                    t_utc = parse_iso_utc(rec.get("time_iso") or rec.get("timestamp") or rec.get("time"))
                    lat = float(rec["latitude"])
                    lon = float(rec["longitude"]) 
                    alt = float(rec.get("altitude_m", rec.get("alt_m", 0.0)))
                    # Sanity check lat/lon
                    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                        continue
                    rows.append(TelemetryRow(t_utc=t_utc, lat=lat, lon=lon, alt_m=alt))
                except Exception:
                    # skip row
                    continue
        if not rows:
            raise ValueError("No valid telemetry rows parsed from CSV")
        return Telemetry(rows)

    def _interp_scalar(self, t: datetime, arr: List[float]) -> float:
        # Assumes self.times sorted, t is timezone-aware UTC
        if t <= self.times[0]:
            return arr[0]
        if t >= self.times[-1]:
            return arr[-1]
        # binary search
        lo, hi = 0, len(self.times) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if t < self.times[mid]:
                hi = mid
            else:
                lo = mid
        t0 = self.times[lo]
        t1 = self.times[hi]
        dt = (t1 - t0).total_seconds()
        w = 0.0 if dt <= 0 else (t - t0).total_seconds() / dt
        return (1 - w) * arr[lo] + w * arr[hi]

    def position_at(self, t: datetime) -> Tuple[float, float, float]:
        """Interpolate lat, lon, alt at UTC time t."""
        lat = self._interp_scalar(t, self.lats)
        lon = self._interp_scalar(t, self.lons)
        alt = self._interp_scalar(t, self.alts)
        return lat, lon, alt

    def azel_at(self, t: datetime, ref_lat: float, ref_lon: float, ref_alt_m: float) -> Tuple[float, float]:
        lat, lon, alt = self.position_at(t)
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt_m)
        return enu_to_azel(e, n, u)
