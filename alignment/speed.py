# alignment/speed.py
"""变速适配器（薄封装 alignment.speed_impl.apply_speed）."""
from __future__ import annotations

from alignment.models import Track
from alignment.parser import serialize_track
# Self-contained speed implementation. Imported from the alignment subpackage
# itself rather than the top-level nodes.py to avoid the ComfyUI
# ``sys.modules['nodes']`` naming conflict.
from alignment.speed_impl import apply_speed as _apply_speed_impl


def apply_speed_change(tracks: list[Track], speed: float) -> list[Track]:
    """对 tracks 应用变速（speed≠1 时）.

    复用 alignment.speed_impl.apply_speed：duration 乘 1/speed，f0 线性插值重采样。

    RF-1: 速度实现现在直接来自 alignment.speed_impl（包内自包含），
    不再延迟 import 外部 nodes.py，避免了 nodes.py 的 g2pM 依赖及模块级
    sys.path 操纵在不同运行时环境（pytest/ComfyUI/直接 import）造成副作用。
    """
    if speed == 1.0:
        return tracks
    track_dicts = [serialize_track(t) for t in tracks]
    result_dicts = _apply_speed_impl(track_dicts, speed)
    return _parse_tracks_from_dicts(result_dicts)


def _parse_tracks_from_dicts(dicts: list[dict]) -> list[Track]:
    """从已解析的 dict 列表构造 Track（避免重新 JSON 序列化）."""
    from alignment.parser import _parse_track
    return [_parse_track(d, i) for i, d in enumerate(dicts)]
