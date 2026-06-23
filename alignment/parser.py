# alignment/parser.py
"""JSON ↔ Token/Track 转换."""
from __future__ import annotations
import json
from alignment.models import Token, Track


def parse_tracks(midi_json_str: str) -> list[Track]:
    """解析 MIDI JSON 字符串为 Track 列表.

    Raises:
        ValueError: JSON 解析失败或字段缺失.
    """
    if not midi_json_str or not midi_json_str.strip():
        raise ValueError("midi_json is empty")
    try:
        data = json.loads(midi_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("midi_json must be a list of track objects")
    return [_parse_track(t, i) for i, t in enumerate(data)]


def _parse_track(track_data: dict, track_idx: int) -> Track:
    """解析单个 track dict."""
    required = ["text", "phoneme", "duration", "note_pitch", "note_type"]
    for field in required:
        if field not in track_data:
            raise ValueError(f"track {track_idx} missing field: {field}")

    texts = track_data["text"].split()
    phonemes = track_data["phoneme"].split()
    durations = track_data["duration"].split()
    pitches = track_data["note_pitch"].split()
    note_types = track_data["note_type"].split()

    n = len(texts)
    if not (len(phonemes) == len(durations) == len(pitches) == len(note_types) == n):
        raise ValueError(
            f"track {track_idx}: field length mismatch "
            f"(text={n}, phoneme={len(phonemes)}, duration={len(durations)}, "
            f"pitch={len(pitches)}, type={len(note_types)})"
        )

    tokens = [
        Token(
            text=texts[i],
            phoneme=phonemes[i],
            duration=float(durations[i]),
            note_pitch=int(pitches[i]),
            note_type=int(note_types[i]),
            index=i,
        )
        for i in range(n)
    ]

    meta = {k: v for k, v in track_data.items()
            if k not in [*required, "f0"]}
    f0 = track_data.get("f0", "")
    return Track(tokens=tokens, meta=meta, f0=f0)


def serialize_track(track: Track) -> dict:
    """把 Track 序列化回 track dict（与原 JSON 格式兼容）."""
    tokens = track.tokens
    result = dict(track.meta)
    result["text"] = " ".join(t.text for t in tokens)
    result["phoneme"] = " ".join(t.phoneme for t in tokens)
    result["duration"] = " ".join(f"{t.duration:.2f}" for t in tokens)
    result["note_pitch"] = " ".join(str(t.note_pitch) for t in tokens)
    result["note_type"] = " ".join(str(t.note_type) for t in tokens)
    if track.f0:
        result["f0"] = track.f0
    return result


def serialize_tracks(tracks: list[Track]) -> str:
    """序列化 Track 列表为 JSON 字符串."""
    return json.dumps([serialize_track(t) for t in tracks],
                      ensure_ascii=False, indent=2)
