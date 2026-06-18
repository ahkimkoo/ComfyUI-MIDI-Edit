# alignment/speed.py
"""变速适配器（薄封装 nodes._apply_speed）."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import _apply_speed

from alignment.models import Track
from alignment.parser import serialize_track


def apply_speed_change(tracks: list[Track], speed: float) -> list[Track]:
    """对 tracks 应用变速（speed≠1 时）.

    复用 nodes._apply_speed：duration 乘 1/speed，f0 线性插值重采样。
    """
    if speed == 1.0:
        return tracks
    track_dicts = [serialize_track(t) for t in tracks]
    result_dicts = _apply_speed(track_dicts, speed)
    return _parse_tracks_from_dicts(result_dicts)


def _parse_tracks_from_dicts(dicts: list[dict]) -> list[Track]:
    """从已解析的 dict 列表构造 Track（避免重新 JSON 序列化）."""
    from alignment.parser import _parse_track
    return [_parse_track(d, i) for i, d in enumerate(dicts)]
