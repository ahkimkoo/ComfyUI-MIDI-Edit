"""SoulX-Singer core functions — ComfyUI-independent.

All public APIs accept/return standard Python types (file paths, numpy arrays,
plain dicts). ComfyUI nodes are thin wrappers around these functions.

Usage from HTTP API server:
    from core.soulsx_singer import transcribe_audio, synthesize_audio
    midi_json = transcribe_audio("/path/to/audio.wav")
    audio_np, sr = synthesize_audio(midi_json, "/path/to/prompt.wav")
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_MODELS_BASE: str | None = None
_svs_model = None
_svs_config = None
_phoneset_path: str | None = None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def set_models_base(path: str) -> None:
    """Override the base models directory (call before first use)."""
    global _MODELS_BASE
    _MODELS_BASE = path


def get_models_base() -> str:
    global _MODELS_BASE
    if _MODELS_BASE is not None:
        return _MODELS_BASE
    try:
        import folder_paths
        _MODELS_BASE = folder_paths.models_dir
    except ImportError:
        _MODELS_BASE = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
    return _MODELS_BASE


def _get_soulxsinger_root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer")


def _get_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_submodule(module_relpath: str):
    """Import a submodule from SoulX-Singer by file path.

    SoulX-Singer's subdirectories (preprocess/, cli/) lack __init__.py and
    cannot be imported with a plain `import` statement. This helper registers
    parent packages in sys.modules so that internal cross-imports work.
    """
    import types
    import importlib

    root = _get_soulxsinger_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    parts = module_relpath.split(".")

    # Ensure all ancestor packages are registered in sys.modules
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name in sys.modules:
            continue
        parent_dir = os.path.join(root, *parts[:i])
        if os.path.isdir(parent_dir):
            mod = types.ModuleType(parent_name)
            mod.__path__ = [parent_dir]
            mod.__package__ = parent_name
            sys.modules[parent_name] = mod

    return importlib.import_module(module_relpath)

    if os.path.isfile(spec_path + ".py"):
        spec_path = spec_path + ".py"

    spec = importlib.util.spec_from_file_location(module_relpath, spec_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_relpath}' from {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_relpath] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Model singletons (lazy)
# ---------------------------------------------------------------------------


def _get_svs_model():
    global _svs_model, _svs_config
    if _svs_model is not None:
        return _svs_model, _svs_config

    file_utils = _load_submodule("soulxsinger.utils.file_utils")
    load_config = file_utils.load_config
    cli_inference = _load_submodule("cli.inference")
    build_model = cli_inference.build_model

    base = get_models_base()
    model_path = os.path.join(base, "Soul-AILab", "SoulX-Singer", "model.pt")
    config_path = os.path.join(_get_soulxsinger_root(), "soulxsinger", "config", "soulxsinger.yaml")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"SoulX-Singer model not found: {model_path}. "
            "Download: hf download Soul-AILab/SoulX-Singer --local-dir "
            f"{os.path.join(base, 'Soul-AILab', 'SoulX-Singer')}"
        )

    config = load_config(config_path)
    device = _get_device()
    _svs_model = build_model(model_path=model_path, config=config, device=device, use_fp16=False)
    _svs_config = config
    return _svs_model, _svs_config


def _get_phoneset_path() -> str:
    global _phoneset_path
    if _phoneset_path is not None:
        return _phoneset_path
    _phoneset_path = os.path.join(_get_soulxsinger_root(), "soulxsinger", "utils", "phoneme", "phone_set.json")
    return _phoneset_path


def _get_preprocess_pipeline(language: str = "Mandarin", save_dir: str | None = None,
                              max_merge_duration: int = 30000):
    pipeline_mod = _load_submodule("preprocess.pipeline")
    PreprocessPipeline = pipeline_mod.PreprocessPipeline

    device = _get_device()
    base = get_models_base()
    pre_base = os.path.join(base, "Soul-AILab", "SoulX-Singer-Preprocess")

    if save_dir is None:
        save_dir = os.path.join(tempfile.gettempdir(), "soulsx_singer_preprocess")

    pipeline = PreprocessPipeline(
        device=device,
        language=language,
        save_dir=save_dir,
        vocal_sep=True,
        max_merge_duration=max_merge_duration,
        midi_transcribe=True,
    )

    sep_dir = os.path.join(pre_base, "mel-band-roformer-karaoke")
    der_dir = os.path.join(pre_base, "dereverb_mel_band_roformer")
    if os.path.isdir(sep_dir) and pipeline.vocal_separator is not None:
        pipeline.vocal_separator.sep_model_path = os.path.join(
            sep_dir, "mel_band_roformer_karaoke_becruily.ckpt"
        )
        pipeline.vocal_separator.sep_config_path = os.path.join(
            sep_dir, "config_karaoke_becruily.yaml"
        )
    if os.path.isdir(der_dir) and pipeline.vocal_separator is not None:
        pipeline.vocal_separator.der_model_path = os.path.join(
            der_dir, "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
        )
        pipeline.vocal_separator.der_config_path = os.path.join(
            der_dir, "dereverb_mel_band_roformer_anvuew.yaml"
        )

    f0_model = os.path.join(pre_base, "rmvpe", "rmvpe.pt")
    if os.path.isfile(f0_model) and pipeline.f0_extractor is not None:
        pipeline.f0_extractor.model_path = f0_model

    if pipeline.vocal_detector is not None:
        pipeline.vocal_detector.cut_wavs_output_dir = os.path.join(save_dir, "cut_wavs")

    zh_asr = os.path.join(pre_base, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    en_asr = os.path.join(pre_base, "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")
    if os.path.isdir(zh_asr) and pipeline.lyric_transcriber is not None:
        pipeline.lyric_transcriber.zh_model_path = zh_asr
    if os.path.isfile(en_asr) and pipeline.lyric_transcriber is not None:
        pipeline.lyric_transcriber.en_model_path = en_asr

    rosvot_model = os.path.join(pre_base, "rosvot", "rosvot", "model.pt")
    rwbd_model = os.path.join(pre_base, "rosvot", "rwbd", "model.pt")
    if os.path.isfile(rosvot_model) and pipeline.note_transcriber is not None:
        pipeline.note_transcriber.rosvot_model_path = rosvot_model
    if os.path.isfile(rwbd_model) and pipeline.note_transcriber is not None:
        pipeline.note_transcriber.rwbd_model_path = rwbd_model

    return pipeline


# ---------------------------------------------------------------------------
# Audio I/O helpers (numpy-based, no ComfyUI types)
# ---------------------------------------------------------------------------


def _ensure_wav_path(audio, sample_rate: int | None, tmpdir: str) -> str:
    """Resolve *audio* to a .wav file path on disk.

    Accepts:
      - str/Path → returned as-is (must exist)
      - (numpy.ndarray, int) tuple → written to a temp wav file
      - numpy.ndarray → requires sample_rate kwarg
    """
    import soundfile as sf

    if isinstance(audio, (str, Path)):
        p = str(audio)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Audio file not found: {p}")
        return p

    if isinstance(audio, tuple):
        arr, sr = audio
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is a numpy array")
        arr, sr = audio, sample_rate
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}. "
                        "Pass a file path, numpy array, or (array, sample_rate) tuple.")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] <= 2:
        arr = arr.T  # (channels, samples) → (samples, channels)
    wav_path = os.path.join(tmpdir, "audio.wav")
    sf.write(wav_path, arr, sr)
    return wav_path


def _load_wav_result(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a wav file and return (waveform, sample_rate).

    waveform shape: (samples,) for mono, (samples, channels) for multi-channel.
    """
    import soundfile as sf
    audio_np, sr = sf.read(wav_path, dtype="float32")
    return audio_np, sr


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def metadata_to_midi_json(metadata_list: list[dict]) -> str:
    """Convert SoulX-Singer metadata list to MIDI JSON string.

    Each metadata dict has space-separated string fields (text, phoneme,
    duration, note_pitch, note_type, f0) and a time field [start_ms, end_ms].
    """
    tracks = []
    for meta in metadata_list:
        track = {}
        for key in ("text", "phoneme", "duration", "note_pitch", "note_type", "f0"):
            val = meta.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                track[key] = " ".join(str(v) for v in val)
            else:
                track[key] = str(val)
        if "text" in track:
            tracks.append(track)
    return json.dumps(tracks, ensure_ascii=False, indent=2)


def midi_json_to_metadata(midi_json_str: str) -> list[dict]:
    """Convert MIDI JSON string to SoulX-Singer metadata list.

    Each track becomes one metadata dict with time inferred from durations.
    """
    tracks = json.loads(midi_json_str)
    if not isinstance(tracks, list):
        raise ValueError("MIDI JSON must be a list of track objects")
    metadata_list = []
    cumulative_time = 0.0
    for track in tracks:
        if "text" not in track:
            continue
        durations = [float(x) for x in track.get("duration", "").split()]
        start_time = cumulative_time
        end_time = cumulative_time + sum(durations) * 1000
        meta = {
            "text": track.get("text", ""),
            "phoneme": track.get("phoneme", ""),
            "duration": track.get("duration", ""),
            "note_pitch": track.get("note_pitch", ""),
            "note_type": track.get("note_type", ""),
            "f0": track.get("f0", ""),
            "time": [int(start_time), int(end_time)],
        }
        metadata_list.append(meta)
        cumulative_time = end_time
    return metadata_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(audio, sample_rate: int | None = None,
                     language: str = "Mandarin",
                     max_merge_duration: int = 30000) -> str:
    """Transcribe audio to MIDI JSON using SoulX-Singer preprocessing.

    Args:
        audio: File path (str/Path), numpy array, or (array, sample_rate) tuple.
        sample_rate: Required only when audio is a bare numpy array.
        language: Lyric language — "Mandarin", "English", or "Cantonese".
        max_merge_duration: Max segment merge duration in ms.

    Returns:
        MIDI JSON string.
    """
    with tempfile.TemporaryDirectory(prefix="soulsx_transcribe_") as tmpdir:
        audio_path = _ensure_wav_path(audio, sample_rate, tmpdir)

        save_dir = os.path.join(tmpdir, "output")
        pipeline = _get_preprocess_pipeline(
            language=language,
            save_dir=save_dir,
            max_merge_duration=max_merge_duration,
        )
        pipeline.run(audio_path=audio_path, language=language)

        meta_path = os.path.join(save_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError("Preprocessing did not produce metadata.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

    return metadata_to_midi_json(metadata_list)


def synthesize_audio(midi_json_str: str, prompt_audio,
                     prompt_sample_rate: int | None = None,
                     control: str = "score",
                     seed: int = 12306,
                     auto_shift: bool = True,
                     pitch_shift: int = 0) -> tuple[np.ndarray, int]:
    """Synthesize singing audio from MIDI JSON and a reference voice.

    Args:
        midi_json_str: MIDI JSON string (target lyrics/notes).
        prompt_audio: Reference voice — file path, numpy array, or (array, sr) tuple.
        prompt_sample_rate: Required only when prompt_audio is a bare numpy array.
        control: "melody" (F0 contour) or "score" (MIDI notes).
        seed: Random seed for reproducibility.
        auto_shift: Auto pitch shift to match reference voice range.
        pitch_shift: Manual pitch shift in semitones (-36 to 36).

    Returns:
        (waveform, sample_rate) — waveform is float32 numpy array.
    """
    import torch
    import soundfile as sf
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, config = _get_svs_model()
    device = _get_device()

    metadata_list = midi_json_to_metadata(midi_json_str)
    if not metadata_list:
        raise ValueError("No valid tracks found in MIDI JSON")

    with tempfile.TemporaryDirectory(prefix="soulsx_synth_") as tmpdir:
        prompt_wav_path = _ensure_wav_path(prompt_audio, prompt_sample_rate, tmpdir)

        cli_inference = _load_submodule("cli.inference")
        svs_process = cli_inference.process

        class Args:
            pass

        args = Args()
        args.device = device
        args.model_path = ""
        args.config = ""
        args.prompt_wav_path = prompt_wav_path
        args.phoneset_path = _get_phoneset_path()
        args.save_dir = os.path.join(tmpdir, "generated")
        args.auto_shift = auto_shift
        args.pitch_shift = pitch_shift
        args.control = control
        args.use_fp16 = False

        prompt_meta_path = os.path.join(tmpdir, "prompt_meta.json")
        target_meta_path = os.path.join(tmpdir, "target_meta.json")

        with open(prompt_meta_path, "w", encoding="utf-8") as f:
            json.dump([metadata_list[0]], f, ensure_ascii=False)
        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False)

        args.prompt_metadata_path = prompt_meta_path
        args.target_metadata_path = target_meta_path

        svs_process(args, config, model)

        generated_path = os.path.join(args.save_dir, "generated.wav")
        if not os.path.isfile(generated_path):
            raise RuntimeError("SVS inference did not produce generated.wav")

    return _load_wav_result(generated_path)
