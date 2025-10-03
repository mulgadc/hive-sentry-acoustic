#!/usr/bin/env python3
"""
hex_array_from_r.py

Compute x,y,z microphone positions for a hexagonal ring given radial distances
and heights, starting at 0° on +X and progressing anti-clockwise in 60° steps.

Outputs a formatted table and a JSON snippet compatible with
catalog/arrays/array_defs.json entries.

Examples
--------
# Six radii (anti-clockwise from 0°), one common height, and an azimuth offset
python scripts/hex_array_from_r.py \
  --radii 0.52 0.52 0.47 0.46 0.48 0.50 \
  --height 0.78 \
  --azimuth-offset-deg 0 \
  --id hex6_ring \
  --labels N NE NW W SW SE

# Per-mic heights instead of common height
python scripts/hex_array_from_r.py \
  --radii 0.52 0.52 0.47 0.46 0.48 0.50 \
  --heights 0.78 0.78 0.78 0.78 0.78 0.78

Notes
-----
- Frame is ENU: x=East, y=North, z=Up.
- Positive rotation for --azimuth-offset-deg is counter-clockwise.
- If you want the first mic to align with +Y (North), use --azimuth-offset-deg 90.
"""
from __future__ import annotations
import argparse
import json
import math
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hex ring coordinates from radii and heights")
    p.add_argument("--radii", nargs=6, type=float, required=True,
                   help="Six radii in meters, anti-clockwise from 0° on +X, spaced 60° apart")
    g_ht = p.add_mutually_exclusive_group(required=True)
    g_ht.add_argument("--height", type=float, help="Single common height (z in meters) for all mics")
    g_ht.add_argument("--heights", nargs=6, type=float,
                      help="Six per-mic heights (z in meters), anti-clockwise order")
    p.add_argument("--azimuth-offset-deg", type=float, default=0.0,
                   help="Global rotation offset in degrees (CCW positive). Applied after base angles.")
    p.add_argument("--id", type=str, default="hex6_ring", help="Array id for JSON snippet")
    p.add_argument("--labels", nargs=6, type=str, default=None,
                   help="Optional six labels for microphones. If omitted, labels are Mic1..Mic6")
    p.add_argument("--save-json", type=str, default=None,
                   help="Optional path to save a JSON file containing this single array definition")
    p.add_argument("--units", type=str, default="meters", help="Units string in JSON output")
    p.add_argument("--frame", type=str, default="ENU", help="Frame string in JSON output")
    return p.parse_args()


def rotate_xy(x: float, y: float, deg: float) -> tuple[float, float]:
    if abs(deg) < 1e-12:
        return x, y
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr


def build_positions(radii: List[float], heights: List[float], az_offset_deg: float) -> List[List[float]]:
    assert len(radii) == 6 and len(heights) == 6
    positions: List[List[float]] = []
    for i in range(6):
        theta_deg = i * 60.0  # base angles: 0, 60, 120, 180, 240, 300
        theta_rad = math.radians(theta_deg)
        r = radii[i]
        x = r * math.cos(theta_rad)
        y = r * math.sin(theta_rad)
        x, y = rotate_xy(x, y, az_offset_deg)
        z = heights[i]
        positions.append([round(x, 6), round(y, 6), round(z, 6)])
    return positions


def format_table(positions: List[List[float]], labels: Optional[List[str]]) -> str:
    lines = []
    header = f"{'Idx':>3}  {'Label':>6}  {'x (m)':>10}  {'y (m)':>10}  {'z (m)':>8}  {'r (m)':>8}  {'theta (deg)':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, (x, y, z) in enumerate(positions, start=1):
        r = math.hypot(x, y)
        th = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        lbl = labels[i-1] if labels else f"Mic{i}"
        lines.append(f"{i:>3}  {lbl:>6}  {x:10.3f}  {y:10.3f}  {z:8.3f}  {r:8.3f}  {th:11.1f}")
    return "\n".join(lines)


def build_json_snippet(array_id: str, positions: List[List[float]], labels: Optional[List[str]],
                       units: str, frame: str) -> dict:
    entry = {
        "version": 1,
        "units": units,
        "frame": frame,
        "arrays": [
            {
                "id": array_id,
                "positions": positions,
                "mic_labels": labels if labels else [f"Mic{i+1}" for i in range(len(positions))],
            }
        ],
    }
    return entry


def main() -> None:
    args = parse_args()
    radii = list(args.radii)
    if args.heights is not None:
        heights = list(args.heights)
    else:
        heights = [float(args.height)] * 6

    positions = build_positions(radii, heights, args.azimuth_offset_deg)

    # Console output
    print("\nHex ring positions (ENU frame; +X=East, +Y=North):")
    print(format_table(positions, args.labels))

    snippet = build_json_snippet(args.id, positions, args.labels, args.units, args.frame)

    print("\nJSON snippet:")
    print(json.dumps(snippet, indent=2))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(snippet, f, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()
