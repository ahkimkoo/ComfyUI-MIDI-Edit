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
    cannot be imported with a plain `import` statement. This helper builds
    each package level from its actual filesystem location so internal
    cross-imports (e.g. ``from preprocess.utils import ...``) resolve correctly.
    """
    import importlib.util

    root = _get_soulxsinger_root()
    parts = module_relpath.split(".")

    for i in range(len(parts)):
        name = ".".join(parts[: i + 1])
        if name in sys.modules:
            continue

        dir_path = os.path.join(root, *parts[: i + 1])
        py_path = dir_path + ".py"

        if i < len(parts) - 1:
            # Intermediate package: must be a directory
            if not os.path.isdir(dir_path):
                raise ImportError(f"Package directory not found: {dir_path}")
            init_path = os.path.join(dir_path, "__init__.py")
            spec = importlib.util.spec_from_file_location(
                name,
                init_path if os.path.isfile(init_path) else None,
                submodule_search_locations=[dir_path],
            )
        else:
            # Leaf module: .py file or package directory
            if os.path.isfile(py_path):
                spec = importlib.util.spec_from_file_location(name, py_path)
            elif os.path.isdir(dir_path):
                init_path = os.path.join(dir_path, "__init__.py")
                spec = importlib.util.spec_from_file_location(
                    name,
                    init_path if os.path.isfile(init_path) else None,
                    submodule_search_locations=[dir_path],
                )
            else:
                raise ImportError(f"Module not found: {name} (looked at {dir_path})")

        if spec is None:
            # Namespace package (no __init__.py) — create a bare module
            import types
            mod = types.ModuleType(name)
            mod.__path__ = [dir_path]
            mod.__package__ = name
        else:
            mod = importlib.util.module_from_spec(spec)

        sys.modules[name] = mod
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(mod)

    return sys.modules[module_relpath]


# ---------------------------------------------------------------------------
# Model singletons (lazy)
# ---------------------------------------------------------------------------


_transformers_patched = False


def _patch_transformers_compat():
    """Monkey-patch for transformers >= 4.53 compatibility with SoulX-Singer.

    In transformers >= 4.53, LlamaAttention.forward() requires position_embeddings.
    SoulX-Singer's LlamaNARDecoderLayer.forward() calls self.self_attn() without
    passing position_embeddings.

    Strategy: patch LlamaAttention.forward() to lazily create its own rotary_emb
    (with matching head_dim) and compute position_embeddings when not provided.
    """
    global _transformers_patched
    if _transformers_patched:
        return

    import importlib.metadata as meta
    try:
        ver = tuple(int(x) for x in meta.version("transformers").split(".")[:2])
    except Exception:
        return

    if ver < (4, 53):
        return

    import torch
    from transformers.models.llama.modeling_llama import LlamaAttention

    _orig_attn_forward = LlamaAttention.forward

    def _patched_attn_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if position_embeddings is None:
            rotary = getattr(self, "rotary_emb", None)
            if rotary is None:
                import inspect
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                head_dim = self.head_dim if hasattr(self, "head_dim") else (
                    self.config.hidden_size // self.config.num_attention_heads
                )
                sig = inspect.signature(LlamaRotaryEmbedding.__init__)
                params = set(sig.parameters.keys()) - {"self"}
                rkwargs = {}
                if "head_dim" in params:
                    rkwargs["head_dim"] = head_dim
                if "max_position_embeddings" in params:
                    rkwargs["max_position_embeddings"] = getattr(self.config, "max_position_embeddings", 4096)
                if "base" in params:
                    rkwargs["base"] = getattr(self.config, "rope_theta", 10000.0)
                if "config" in params and not rkwargs:
                    rkwargs["config"] = self.config
                rotary = LlamaRotaryEmbedding(**rkwargs)
                rotary = rotary.to(hidden_states.device, dtype=hidden_states.dtype)
                self.rotary_emb = rotary
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                ).unsqueeze(0)
            position_embeddings = rotary(hidden_states, position_ids)
        result = _orig_attn_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # SoulX-Singer expects 3-tuple: (hidden_states, attn_weights, present_key_value)
        # Newer transformers may return 2-tuple or a different format
        if isinstance(result, tuple) and len(result) == 2:
            return (result[0], result[1], None)
        return result

    LlamaAttention.forward = _patched_attn_forward

    # Also patch LlamaConfig to ensure _attn_implementation is set.
    # SoulX-Singer creates LlamaConfig with positional args only, leaving
    # _attn_implementation as None, which breaks transformers >= 4.57.
    from transformers import LlamaConfig as _LlamaCfg
    _orig_cfg_init = _LlamaCfg.__init__
    def _patched_cfg_init(self, *args, **kwargs):
        _orig_cfg_init(self, *args, **kwargs)
        if getattr(self, "_attn_implementation", None) is None:
            self._attn_implementation = "eager"
    _LlamaCfg.__init__ = _patched_cfg_init

    _transformers_patched = True
    print(f"[MIDI-Edit] Patched LlamaAttention.forward for transformers {ver} compatibility")


def _get_svs_model():
    global _svs_model, _svs_config
    if _svs_model is not None:
        return _svs_model, _svs_config

    _patch_transformers_compat()

    file_utils = _load_submodule("soulxsinger.utils.file_utils")
    load_config = file_utils.load_config
    cli_inference = _load_submodule("cli.inference")
    build_model = cli_inference.build_model

    print(f"[MIDI-Edit] Building SVS model...")

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
    # Build in FP32 to match the reference SVS implementation, which runs pure
    # FP32 (no model.half()). FP16 weights are a secondary suspect for degraded
    # audio quality / garbled pronunciation. FP32 is the safe default; per-call
    # mixed precision can still be requested via synthesize_audio(use_fp16=True),
    # which uses autocast on the fly without changing the model dtype.
    use_fp16 = False
    print(f"[MIDI-Edit] Loading SVS model on {device}, fp16={use_fp16}")
    _svs_model = build_model(model_path=model_path, config=config, device=device, use_fp16=use_fp16)
    _svs_config = config
    return _svs_model, _svs_config


def _get_phoneset_path() -> str:
    global _phoneset_path
    if _phoneset_path is not None:
        return _phoneset_path
    _phoneset_path = os.path.join(_get_soulxsinger_root(), "soulxsinger", "utils", "phoneme", "phone_set.json")
    return _phoneset_path


def _ensure_pretrained_models_links():
    """Create pretrained_models/ symlinks inside SoulX-Singer.

    SoulX-Singer's code uses relative paths like
    ``pretrained_models/SoulX-Singer-Preprocess/...``. The official setup
    downloads models into ``pretrained_models/`` inside the repo. Since our
    models live in ComfyUI's models/ directory, we create symlinks so the
    relative paths resolve without downloading anything.

    Only symlinks are created (under pretrained_models/), no SoulX-Singer
    source code is modified.
    """
    sx_root = _get_soulxsinger_root()
    base = get_models_base()

    pairs = [
        (
            os.path.join(sx_root, "pretrained_models", "SoulX-Singer"),
            os.path.join(base, "Soul-AILab", "SoulX-Singer"),
        ),
        (
            os.path.join(sx_root, "pretrained_models", "SoulX-Singer-Preprocess"),
            os.path.join(base, "Soul-AILab", "SoulX-Singer-Preprocess"),
        ),
    ]

    for link_path, target_path in pairs:
        if os.path.islink(link_path):
            continue
        if os.path.isdir(link_path):
            continue
        if not os.path.isdir(target_path):
            continue
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(target_path, link_path)


def _ensure_nltk_for_preprocess():
    """Ensure NLTK data is available before the SoulX-Singer preprocess pipeline runs.

    The preprocess chain's English g2p (``SoulX-Singer/preprocess/tools/g2p.py``,
    which we must not modify) calls ``nltk.pos_tag`` at *pipeline run* time, and
    that requires NLTK resources such as ``averaged_perceptron_tagger_eng``.

    Those resources are downloaded lazily by :mod:`core.g2p`, but only when the
    MIDIEdit lyrics path (:func:`core.g2p._get_g2p_en`) is taken — the
    transcription path never touches it, so ``ComfyUI/models/nltk`` would be
    empty and ``LookupError`` would be raised. Even after download, the env-var
    ``NLTK_DATA`` setdefault in :mod:`core.g2p` is ineffective once nltk has been
    imported (which the preprocess chain does), so we must also register the
    directory on ``nltk.data.path`` explicitly.

    Reuses :func:`core.g2p.ensure_nltk_data` (single source of truth: path
    injection + download into ``ComfyUI/models/nltk``) and must run *before*
    :func:`_load_submodule` so any nltk initialization in the imported modules
    already sees the registered path.
    """
    from core.g2p import ensure_nltk_data

    ensure_nltk_data()


def _get_preprocess_pipeline(language: str = "Mandarin", save_dir: str | None = None,
                              max_merge_duration: int = 30000):
    # Must run before _load_submodule("preprocess.*"): the preprocess chain
    # imports g2p_en (-> nltk), so register ComfyUI/models/nltk on nltk.data.path
    # and ensure required packages are downloaded before any nltk init sees it.
    _ensure_nltk_for_preprocess()

    _ensure_pretrained_models_links()

    pipeline_mod = _load_submodule("preprocess.pipeline")
    PreprocessPipeline = pipeline_mod.PreprocessPipeline

    device = _get_device()
    print(f"[MIDI-Edit] Preprocess pipeline using device: {device}")

    if save_dir is None:
        save_dir = os.path.join(tempfile.gettempdir(), "soulsx_singer_preprocess")

    # SoulX-Singer uses relative paths from its repo root.
    base = get_models_base()
    pre_base = os.path.join(base, "Soul-AILab", "SoulX-Singer-Preprocess")
    sx_root = _get_soulxsinger_root()
    orig_cwd = os.getcwd()
    try:
        os.chdir(sx_root)
        pipeline = PreprocessPipeline(
            device=device,
            language=language,
            save_dir=save_dir,
            vocal_sep=True,
            max_merge_duration=max_merge_duration,
            midi_transcribe=True,
        )
    finally:
        os.chdir(orig_cwd)

    # Override to absolute paths for components that need them at runtime
    if pipeline.vocal_separator is not None:
        sep = os.path.join(pre_base, "mel-band-roformer-karaoke", "mel_band_roformer_karaoke_becruily.ckpt")
        if os.path.isfile(sep):
            pipeline.vocal_separator.sep_model_path = sep
            pipeline.vocal_separator.sep_config_path = os.path.join(
                pre_base, "mel-band-roformer-karaoke", "config_karaoke_becruily.yaml")
        der = os.path.join(pre_base, "dereverb_mel_band_roformer", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
        if os.path.isfile(der):
            pipeline.vocal_separator.der_model_path = der
            pipeline.vocal_separator.der_config_path = os.path.join(
                pre_base, "dereverb_mel_band_roformer", "dereverb_mel_band_roformer_anvuew.yaml")

    f0 = os.path.join(pre_base, "rmvpe", "rmvpe.pt")
    if os.path.isfile(f0) and pipeline.f0_extractor is not None:
        pipeline.f0_extractor.model_path = f0

    if pipeline.vocal_detector is not None:
        pipeline.vocal_detector.cut_wavs_output_dir = os.path.join(save_dir, "cut_wavs")

    zh_asr = os.path.join(pre_base, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    en_asr = os.path.join(pre_base, "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")
    if os.path.isdir(zh_asr) and pipeline.lyric_transcriber is not None:
        pipeline.lyric_transcriber.zh_model_path = zh_asr
    if os.path.isfile(en_asr) and pipeline.lyric_transcriber is not None:
        pipeline.lyric_transcriber.en_model_path = en_asr

    rosvot = os.path.join(pre_base, "rosvot", "rosvot", "model.pt")
    rwbd = os.path.join(pre_base, "rosvot", "rwbd", "model.pt")
    if os.path.isfile(rosvot) and pipeline.note_transcriber is not None:
        pipeline.note_transcriber.rosvot_model_path = rosvot
    if os.path.isfile(rwbd) and pipeline.note_transcriber is not None:
        pipeline.note_transcriber.rwbd_model_path = rwbd

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
    Always returns at least 1D array.
    """
    import soundfile as sf
    audio_np, sr = sf.read(wav_path, dtype="float32")
    if audio_np.ndim == 0:
        audio_np = audio_np.reshape(1)
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


def _preprocess_audio_to_metadata(audio, sample_rate: int | None = None,
                                  language: str = "Mandarin",
                                  max_merge_duration: int = 30000) -> list[dict]:
    """Run SoulX-Singer preprocessing on *audio* and return the raw metadata list.

    This is the shared core of :func:`transcribe_audio`. It reuses the existing
    preprocess pipeline (vocal separation, F0 extraction, VAD, lyrics ASR, note
    transcription) without re-implementing any of it, and returns the raw
    metadata list (one dict per segment) exactly as written by the pipeline —
    **not** converted to MIDI JSON.

    The returned metadata describes the *actual acoustic content* of *audio*
    (phoneme / duration / note_pitch / note_type / f0 aligned to the waveform),
    which is the contract SoulX-Singer requires for prompt metadata
    (see :func:`synthesize_audio`).
    """
    with tempfile.TemporaryDirectory(prefix="soulsx_preprocess_") as tmpdir:
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

        if not metadata_list:
            raise RuntimeError(
                "Preprocessing produced an empty metadata list; "
                "check that the audio actually contains vocals."
            )

    return metadata_list


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
    metadata_list = _preprocess_audio_to_metadata(
        audio, sample_rate=sample_rate, language=language,
        max_merge_duration=max_merge_duration,
    )
    return metadata_to_midi_json(metadata_list)


def _validate_metadata_items(metadata_list) -> list[dict]:
    """Ensure *metadata_list* is a list of non-empty dicts.

    Raises ``ValueError`` if any element is not a ``dict`` or is an empty ``dict``.
    A non-empty list with invalid content is a user error; an empty list itself
    is intentionally allowed (semantics: "not provided" -> auto-preprocess).
    """
    for item in metadata_list:
        if not isinstance(item, dict) or not item:
            kind = "empty dict" if isinstance(item, dict) else type(item).__name__
            raise ValueError(
                "prompt_metadata list must contain only non-empty dict items; "
                f"found element of type {kind}"
            )
    return metadata_list


def _coerce_prompt_metadata(prompt_metadata) -> list[dict]:
    """Coerce a user-supplied *prompt_metadata* value into a metadata list.

    Accepts:
      - ``None`` / empty string  → ``[]`` (caller falls back to auto-preprocess)
      - MIDI JSON string (our track format: has ``text``, no ``time``)
        → converted via :func:`midi_json_to_metadata`
      - JSON string of a raw metadata list (dicts carry a ``time`` field)
        → ``json.loads`` used directly
      - ``list`` → used directly (each item must be a non-empty dict)
      - ``dict`` → wrapped into a single-element list

    The result always describes the prompt waveform (the reference voice), never
    the target content.

    Format disambiguation for JSON strings: the ``time`` / ``text`` fields tell
    the two accepted formats apart. Our MIDI JSON *track* format (produced by
    :func:`metadata_to_midi_json` / the ``MIDI Transcribe Audio`` node) uses
    space-separated string fields and always carries a ``text`` key but **no**
    ``time`` key, so it must pass through :func:`midi_json_to_metadata` to infer
    timing. Raw SoulX-Singer metadata dicts (as written by the preprocess
    pipeline and by :func:`_preprocess_audio_to_metadata`) carry a ``time``
    ``[start_ms, end_ms]`` field and are already the native format, so they are
    used as-is. This rule is based on the current contract of these two formats.
    """
    if prompt_metadata is None:
        return []
    if isinstance(prompt_metadata, list):
        return _validate_metadata_items(prompt_metadata)
    if isinstance(prompt_metadata, dict):
        return [prompt_metadata]
    if isinstance(prompt_metadata, str):
        s = prompt_metadata.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"prompt_metadata is not valid JSON: {e}") from e
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            # Distinguish our MIDI JSON track format (string fields, "text",
            # no "time") from a raw metadata list (carries "time"). Only the
            # former needs midi_json_to_metadata; the latter is already the
            # native SoulX-Singer metadata format.
            if (parsed and isinstance(parsed[0], dict)
                    and "time" not in parsed[0] and "text" in parsed[0]):
                return midi_json_to_metadata(s)
            return _validate_metadata_items(parsed)
        raise ValueError("prompt_metadata JSON must be a list or object")
    raise TypeError(
        f"Unsupported prompt_metadata type: {type(prompt_metadata)}"
    )


def _resolve_prompt_metadata(prompt_metadata, fallback_audio,
                             fallback_sample_rate: int | None = None,
                             fallback_language: str = "Mandarin") -> list[dict]:
    """Return prompt metadata that matches the reference (prompt) audio.

    Resolution order:
      1. If *prompt_metadata* is provided and non-empty (bypass), coerce it to a
         metadata list. This is typically the output of running
         :func:`transcribe_audio` (or the ``MIDI Transcribe Audio`` node) on the
         prompt audio.
      2. Otherwise, auto-preprocess *fallback_audio* to obtain metadata that
         describes its real acoustic content.

    SoulX-Singer requires the prompt metadata's phoneme / duration / note_pitch /
    note_type / f0 to match the prompt waveform. The reference voice is a
    different recording from the target, so prompt metadata must NEVER be derived
    from the target MIDI JSON.
    """
    if prompt_metadata is not None:
        coerced = _coerce_prompt_metadata(prompt_metadata)
        if coerced:
            return coerced
        # Blank/empty -> treat as "not provided" and auto-preprocess.
    return _preprocess_audio_to_metadata(
        fallback_audio,
        sample_rate=fallback_sample_rate,
        language=fallback_language,
    )


def synthesize_audio(midi_json_str: str, prompt_audio,
                     prompt_sample_rate: int | None = None,
                     prompt_metadata=None,
                     prompt_language: str = "Mandarin",
                     control: str = "melody",
                     seed: int = 12306,
                     auto_shift: bool = True,
                     pitch_shift: int = 0,
                     use_fp16: bool = False,
                     cfg: float | None = None,
                     n_steps: int | None = None) -> tuple[np.ndarray, int]:
    """Synthesize singing audio from MIDI JSON and a reference voice.

    Args:
        midi_json_str: MIDI JSON string (the TARGET lyrics/notes to sing).
        prompt_audio: Reference voice — file path, numpy array, or (array, sr)
            tuple. Provides the target timbre; its waveform is paired with the
            prompt metadata.
        prompt_sample_rate: Required only when prompt_audio is a bare numpy array.
        prompt_metadata: Optional prompt metadata describing the prompt audio's
            real acoustic content (phoneme/duration/note_pitch/note_type/f0).
            Recommended: feed the output of ``MIDI Transcribe Audio`` run on the
            prompt audio here, to avoid re-running preprocessing each call. If
            omitted/empty, the prompt audio is preprocessed internally.
        prompt_language: Language used when auto-preprocessing the prompt audio.
        control: "melody" (F0 contour) or "score" (MIDI note pitches).
        seed: Random seed for reproducibility.
        auto_shift: Auto pitch shift to match reference voice range.
        pitch_shift: Manual pitch shift in semitones (-36 to 36).
        use_fp16: If True, run inference with autocast mixed precision on CUDA
            (faster). Defaults to False (pure FP32) to match the reference SVS
            implementation and avoid quality regressions.
        cfg: Classifier-free guidance scale (default 3 when None, read from the
            SVS config). Higher values enforce stronger adherence to the lyrics /
            phoneme content — in ``melody`` mode, raise toward 4-5 if diction is
            muddy / swallowed; too high may cause over-saturation or artifacts.
        n_steps: Number of flow-matching reverse-diffusion steps (default 32 when
            None). Raising it slightly improves quality at the cost of inference
            speed; lowering below 16 trades quality for speed.

    Note:
        ``rescale_cfg`` is intentionally **not** exposed: upstream
        ``SoulXSinger.infer`` does not pass it through (it is hardcoded to 0.75
        inside ``flow_matching.reverse_diffusion``). Exposing it would require
        monkey-patching model internals, which violates the reuse principle.

    Returns:
        (waveform, sample_rate) — waveform is float32 numpy array.
    """
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, config = _get_svs_model()
    device = _get_device()

    metadata_list = midi_json_to_metadata(midi_json_str)
    if not metadata_list:
        raise ValueError("No valid tracks found in MIDI JSON")

    # Resolve prompt metadata so it ALWAYS matches the reference (prompt) audio.
    # Previously the first TARGET track was (incorrectly) used as the prompt,
    # causing the prompt waveform to be truncated/misaligned to target content
    # (data_processor truncates waveform to min_frame*hop_size derived from the
    # metadata's duration/f0), producing garbled pronunciation.
    prompt_meta_list = _resolve_prompt_metadata(
        prompt_metadata,
        fallback_audio=prompt_audio,
        fallback_sample_rate=prompt_sample_rate,
        fallback_language=prompt_language,
    )
    if not prompt_meta_list:
        raise ValueError(
            "Prompt metadata is empty. Provide prompt_metadata (e.g. the output "
            "of MIDI Transcribe Audio on the prompt audio) or a valid prompt_audio."
        )

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
        args.use_fp16 = use_fp16

        prompt_meta_path = os.path.join(tmpdir, "prompt_meta.json")
        target_meta_path = os.path.join(tmpdir, "target_meta.json")

        with open(prompt_meta_path, "w", encoding="utf-8") as f:
            json.dump(prompt_meta_list, f, ensure_ascii=False)
        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False)

        args.prompt_metadata_path = prompt_meta_path
        args.target_metadata_path = target_meta_path

        # Build an EFFECTIVE config so per-call inference knobs (cfg, n_steps)
        # can be overridden WITHOUT mutating the global _svs_config singleton.
        # Upstream `cli.inference.process` reads n_steps/cfg from config.infer
        # (NOT from args), so we deep-copy the config and override on the copy.
        # OmegaConf.create(to_yaml(...)) yields a structurally independent copy;
        # subsequent calls that pass cfg/n_steps=None keep using the defaults
        # (3 / 32). rescale_cfg is NOT overridden here: SoulXSinger.infer does
        # not accept it (hardcoded 0.75 in flow_matching.reverse_diffusion).
        from omegaconf import OmegaConf
        eff_config = OmegaConf.create(OmegaConf.to_yaml(config))
        if cfg is not None:
            eff_config.infer.cfg = cfg
        if n_steps is not None:
            eff_config.infer.n_steps = n_steps

        svs_process(args, eff_config, model)

        generated_path = os.path.join(args.save_dir, "generated.wav")
        if not os.path.isfile(generated_path):
            raise RuntimeError("SVS inference did not produce generated.wav")

        return _load_wav_result(generated_path)
