# alignment/speed_impl.py
"""Speed adjustment implementation (self-contained).

Extracted from ``nodes._apply_speed`` so the alignment subpackage does not
import the external ``nodes.py`` (which clashes with ComfyUI core's
``nodes.py`` under ``sys.modules['nodes']``).

Behaviour is identical to ``nodes._apply_speed`` + its private formatting
helpers (``_fmt_durs`` / ``_fmt_f0``).
"""
from __future__ import annotations


def apply_speed(midi_data: list, speed: float) -> list:
    """Apply speed change to all tracks: scale durations and resample f0.

    Args:
        midi_data: List of track dicts (already processed by replace_lyrics).
        speed: Speed multiplier (e.g. 1.5 = 150% speed = faster = shorter duration,
               0.5 = 50% speed = slower = longer duration).
               duration_new = duration_orig / speed

    Returns:
        Modified midi_data with scaled durations and resampled f0.
    """
    import numpy as _np

    ratio = 1.0 / speed  # duration scale factor (speed up -> ratio < 1 -> shorter)

    for track in midi_data:
        # Scale durations
        if "duration" in track:
            dur_vals = [float(x) * ratio for x in track["duration"].split(" ")]
            track["duration"] = " ".join(_fmt_durs(dur_vals))

        # Scale time range (used by downstream to preallocate audio buffer)
        if "time" in track and isinstance(track["time"], list) and len(track["time"]) == 2:
            track["time"] = [round(track["time"][0] * ratio), round(track["time"][1] * ratio)]

        # Resample f0 (frame-level data at ~50fps)
        if "f0" in track and track["f0"].strip():
            f0_vals = [float(x) for x in track["f0"].split(" ")]
            orig_len = len(f0_vals)
            new_len = max(1, round(orig_len * ratio))

            if new_len == orig_len:
                # No change needed
                track["f0"] = " ".join(_fmt_f0(v) for v in f0_vals)
            elif new_len > orig_len:
                # Stretch: linear interpolation
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(_fmt_f0(v) for v in resampled)
            else:
                # Shrink: linear interpolation then take fewer samples
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(_fmt_f0(v) for v in resampled)

    return midi_data


def _fmt_durs(durations: list[float]) -> list[str]:
    """Format a list of durations to 2 decimal places, adjusting the last
    element so the rounded total matches the true total."""
    if not durations:
        return []
    true_total = sum(durations)
    rounded = [round(d, 2) for d in durations]
    rounded_total = sum(rounded)
    diff = round(true_total - rounded_total, 2)
    if diff != 0 and rounded:
        rounded[-1] = round(rounded[-1] + diff, 2)
    return [f"{d:.2f}" for d in rounded]


def _fmt_f0(v) -> str:
    """Format an f0 value, cleaning up float artifacts."""
    f = float(v)
    if f == 0.0:
        return "0.0"
    s = f"{f:.1f}".rstrip("0").rstrip(".")
    return s if s else "0"
