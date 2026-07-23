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
import re
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


# ---------------------------------------------------------------------------
# Reference-lyrics-biased ASR wrapper
# ---------------------------------------------------------------------------


# Regex used to normalize reference lyrics: keep only CJK chars + ASCII letters.
# Punctuation, whitespace, digits etc. are stripped before passing to ASR.
_REF_LYRICS_STRIP_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z]")


class _ForceAlignLyricTranscriber:
    """Wrap a SoulX-Singer ``LyricTranscriber`` for two jobs:

    1. **Optional force-alignment** with user-provided reference lyrics (see
       :func:`transcribe_audio` ``reference_lyrics``).
    2. **Always capture** the per-segment ``words`` list — the *pre-ROSVOT*
       lyrics that get fed into ``note_transcriber``. The captured words are
       exposed via :attr:`captured_segment_words` so callers can build a
       lyrics-text output alongside the MIDI JSON.

    Capturing is necessary because once ROSVOT runs, the text gets expanded
    with melisma duplications (one char across multiple notes) that are NOT
    in the source lyrics. The pre-ROSVOT ``words`` is the right artifact for
    "what lyrics were detected / forced" before ROSVOT's musical expansion.

    Force-alignment strategy (per segment), only when ``reference_lyrics`` is
    non-empty:

    1. **First-pass ASR** (no hotword) → rough char count for this segment.
    2. **Slice reference** by char count: take next *N* chars from remaining
       reference, where *N* = ASR's char count for this segment.
    3. **Second-pass ASR** with ``hotword=<reference phrases>``: Paraformer
       biased toward reference, fixes most phonetic confusions.
    4. **DTW post-correction**: char-level Dynamic Time Warping between ASR
       output and reference slice. Guarantees text matches reference exactly
       while preserving ASR's char-level timestamps.

    The wrapper preserves the original ``process(wav_fn, language)``
    signature so ``pipeline.run()`` uses it transparently. SPs / word_durs /
    f0 post-processing are still driven by audio; only the *identity* of each
    word is replaced.

    Pitch / duration / f0 / melisma detection (ROSVOT) are NOT affected.
    """

    # Paraformer hotword phrases longer than this are less effective.
    # 5 chars per phrase empirically works well for Mandarin lyrics.
    _HOTWORD_PHRASE_LEN = 5

    def __init__(self, inner, reference_lyrics: str | None = None):
        """Args:
            inner: the original ``LyricTranscriber`` instance (must already be
                initialised with the Paraformer model).
            reference_lyrics: optional user-provided lyrics. Punctuation and
                whitespace are stripped; only CJK + ASCII letters are kept.
                When empty/None, the wrapper delegates to the inner
                transcriber and only does lyrics capture.
        """
        self.inner = inner
        self.reference_chars = _REF_LYRICS_STRIP_RE.sub("", reference_lyrics or "")
        # Position counter consumed across segments (one transcribe_audio call).
        self._ref_pos = 0
        # Per-segment diagnostics, exposed for warning messages.
        self.diagnostics: list[dict] = []
        # CAPTURE: per-segment pre-ROSVOT word lists (incl. <SP>). Populated
        # by every process() call. Used to build the lyrics_text output.
        self.captured_segment_words: list[list[str]] = []

    # Forward attributes that `_get_preprocess_pipeline` mutates after wrapping
    # (zh_model_path / en_model_path) so the path-override block still works.
    @property
    def zh_model_path(self):
        return self.inner.zh_model_path

    @zh_model_path.setter
    def zh_model_path(self, v):
        self.inner.zh_model_path = v

    @property
    def en_model_path(self):
        return self.inner.en_model_path

    @en_model_path.setter
    def en_model_path(self, v):
        self.inner.en_model_path = v

    # Verbose / device / verbose passthroughs for `pipeline.run` logging.
    @property
    def verbose(self):
        return self.inner.verbose

    @property
    def device(self):
        return self.inner.device

    def process(self, wav_fn: str, language: str | None = "Mandarin"):
        """Drop-in replacement for ``LyricTranscriber.process``.

        Always captures the returned ``words`` list into
        :attr:`captured_segment_words` for downstream lyrics-text extraction.

        Force-alignment is skipped (delegating to inner) when:
          - reference lyrics is empty, or
          - all reference chars have been consumed by earlier segments, or
          - language is English (NeMo Parakeet does not accept hotword the same
            way Paraformer-zh does; English path is left untouched for now).
        """
        lang = (language or "auto").lower()
        is_zh_path = lang in {"mandarin", "cantonese", "zh", "中文", ""}
        if (not self.reference_chars
                or self._ref_pos >= len(self.reference_chars)
                or not is_zh_path):
            words, word_durs = self.inner.process(wav_fn, language)
            self.captured_segment_words.append(list(words))
            if self.reference_chars and self._ref_pos >= len(self.reference_chars):
                print(f"[MIDI-Edit] ForceAlign: WARNING — reference exhausted "
                      f"(pos={self._ref_pos}/{len(self.reference_chars)}), "
                      f"segment {os.path.basename(wav_fn)} falls back to "
                      f"default ASR (no DTW correction)")
            return words, word_durs

        # Load the shared transcription utilities lazily (they live in the
        # SoulX-Singer submodule, which is only importable after
        # `_load_submodule("preprocess.tools.lyric_transcription")`).
        from preprocess.tools.lyric_transcription import (
            _build_words_with_gaps, _word_dur_post_process,
        )

        # ---- Step 1: first-pass ASR (no hotword) → rough char count ----
        # Used ONLY to generate the hotword; the actual reference slice uses
        # the second-pass ASR's count (more accurate thanks to hotword bias).
        out_first = self.inner.zh_model.model.generate(
            wav_fn, output_timestamp=True,
        )[0]
        first_words = out_first["text"].replace("@", "").split(" ")
        n_chars_first = len(first_words)
        print(f"[MIDI-Edit] ForceAlign seg={os.path.basename(wav_fn)}: "
              f"first-pass ASR ({n_chars_first} chars): {' '.join(first_words)}")

        # ---- Step 2: slice reference for HOTWORD ONLY (rough) ----
        # DON'T advance ref_pos yet — we'll re-slice after second-pass.
        seg_ref_for_hotword = self.reference_chars[
            self._ref_pos:self._ref_pos + n_chars_first
        ]
        if not seg_ref_for_hotword:
            words, word_durs = self.inner.process(wav_fn, language)
            self.captured_segment_words.append(list(words))
            return words, word_durs

        # ---- Step 3: second-pass ASR biased with hotword ----
        hotword = " ".join(self._split_phrases(seg_ref_for_hotword))
        try:
            out = self.inner.zh_model.model.generate(
                wav_fn, output_timestamp=True, hotword=hotword,
            )[0]
        except Exception as e:  # hotword not supported / model error
            print(f"[MIDI-Edit] ForceAlign: hotword ASR failed ({e}); "
                  f"falling back to first-pass")
            out = out_first

        asr_words = out["text"].replace("@", "").split(" ")
        asr_ts = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]
        n_chars_second = len(asr_words)
        print(f"[MIDI-Edit] ForceAlign seg={os.path.basename(wav_fn)}: "
              f"second-pass ASR ({n_chars_second} chars): {' '.join(asr_words)}")

        # ---- Step 4: RE-SLICE reference using second-pass char count ----
        # Use second-pass count (with hotword) for the slice — more accurate
        # than first-pass. No overestimate: overeating ref chars from future
        # segments causes cascading offset errors (worse than undercounting).
        seg_ref = self.reference_chars[
            self._ref_pos:self._ref_pos + n_chars_second
        ]
        self._ref_pos += len(seg_ref)
        print(f"[MIDI-Edit] ForceAlign seg={os.path.basename(wav_fn)}: "
              f"ref slice [{self._ref_pos - len(seg_ref)}:{self._ref_pos}] "
              f"({len(seg_ref)} chars): {seg_ref}")

        # ---- Step 5: DTW ----
        final_words = self._dtw_align(list(seg_ref), asr_words)
        diffs = [(a, b) for a, b in zip(asr_words, final_words) if a != b]
        print(f"[MIDI-Edit] ForceAlign seg={os.path.basename(wav_fn)}: "
              f"DTW output ({len(final_words)} chars): {' '.join(final_words)}"
              + (f"  diffs={diffs}" if diffs else "  (no diffs)"))

        # Record diagnostics for the wrapper's caller.
        self.diagnostics.append({
            "wav_fn": wav_fn,
            "hotword": hotword,
            "asr_words": asr_words,
            "final_words": final_words,
            "diffs": diffs,
            "ref_slice": seg_ref,
        })

        # ---- Step 6: build (words, word_durs) identical to original ASR ----
        words, word_durs = _build_words_with_gaps(final_words, asr_ts, wav_fn)
        f0_path = os.path.splitext(wav_fn)[0] + "_f0.npy"
        if os.path.exists(f0_path):
            words, word_durs = _word_dur_post_process(
                words, word_durs, np.load(f0_path),
            )
        self.captured_segment_words.append(list(words))
        return words, word_durs

    @classmethod
    def _split_phrases(cls, text: str) -> list[str]:
        """Split a reference slice into ~5-char phrases for hotword biasing."""
        n = cls._HOTWORD_PHRASE_LEN
        return [text[i:i + n] for i in range(0, len(text), n)]

    @staticmethod
    def _dtw_align(ref_chars: list[str], asr_words: list[str]) -> list[str]:
        """Char-level DTW mapping each ASR position to its best ref char.

        - 'sub' (1 ref ↔ 1 asr): substitute asr with ref
        - 'ins' (1 ref ↔ 0 asr): ref char dropped (no ASR position to carry it)
        - 'del' (0 ref ↔ 1 asr): asr is "extra"; assign it the nearest ref char

        The output always has ``len == len(asr_words)`` so ASR's structure
        (including melisma-induced duplications that ROSVOT will later expand)
        is preserved — only the *identity* of each word changes.

        Time complexity is O(M*N) where M, N are small (per-segment counts).
        """
        M, N = len(ref_chars), len(asr_words)
        if N == 0:
            return []
        if M == 0:
            return list(asr_words)

        INF = float("inf")
        dp = [[INF] * (N + 1) for _ in range(M + 1)]
        back = [[None] * (N + 1) for _ in range(M + 1)]
        dp[0][0] = 0
        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    cost = 0 if ref_chars[i - 1] == asr_words[j - 1] else 1
                    if dp[i - 1][j - 1] + cost < dp[i][j]:
                        dp[i][j] = dp[i - 1][j - 1] + cost
                        back[i][j] = ("sub", i - 1, j - 1)
                if i > 0:
                    if dp[i - 1][j] + 1 < dp[i][j]:
                        dp[i][j] = dp[i - 1][j] + 1
                        back[i][j] = ("ins", i - 1, j)
                if j > 0:
                    if dp[i][j - 1] + 1 < dp[i][j]:
                        dp[i][j] = dp[i][j - 1] + 1
                        back[i][j] = ("del", i, j - 1)

        ops = []
        i, j = M, N
        while i > 0 or j > 0:
            op, pi, pj = back[i][j]
            ops.append((op, pi, pj))
            i, j = pi, pj
        ops.reverse()

        new_words: list[str | None] = [None] * N
        for op, pi, pj in ops:
            if op == "sub":
                new_words[pj] = ref_chars[pi]
            elif op == "del":
                # Extra ASR position: assign nearest ref char so ROSVOT still
                # has a meaningful word for every note it detects.
                new_words[pj] = ref_chars[min(pi, M - 1)]
        for k in range(N):
            if new_words[k] is None:
                new_words[k] = asr_words[k]
        return new_words  # type: ignore[return-value]

    @staticmethod
    def _dtw_align_with_insert(
        ref_chars: list[str],
        asr_words: list[str],
        asr_ts: list[list[float]],
    ) -> tuple[list[str], list[list[float]], int]:
        """DTW that INSERTS missing ref chars (instead of dropping them).

        When ASR detects fewer chars than the reference (e.g., ASR missed a
        held note), the 'ins' ops INSERT the missing ref char by splitting
        the adjacent ASR char's timestamp. This ensures the output text
        matches the reference exactly, even when ASR undercounts.

        Only 'ins' ops WITHIN 1 position of the last 'sub' are inserted
        (they represent chars ASR missed in this segment). 'ins' ops further
        away are NOT inserted (they belong to future segments).

        Returns:
            (out_words, out_ts, consumed_ref_count)
            - out_words: aligned char list (may be longer than asr_words)
            - out_ts: timestamps for each output char
            - consumed_ref_count: how many ref chars this segment consumed
              (for advancing ref_pos)
        """
        M, N = len(ref_chars), len(asr_words)
        if N == 0:
            return [], [], 0
        if M == 0:
            return list(asr_words), asr_ts, 0

        # ---- Standard DTW to find alignment ops ----
        INF = float("inf")
        dp = [[INF] * (N + 1) for _ in range(M + 1)]
        back = [[None] * (N + 1) for _ in range(M + 1)]
        dp[0][0] = 0
        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    cost = 0 if ref_chars[i - 1] == asr_words[j - 1] else 1
                    if dp[i - 1][j - 1] + cost < dp[i][j]:
                        dp[i][j] = dp[i - 1][j - 1] + cost
                        back[i][j] = ("sub", i - 1, j - 1)
                if i > 0:
                    if dp[i - 1][j] + 1 < dp[i][j]:
                        dp[i][j] = dp[i - 1][j] + 1
                        back[i][j] = ("ins", i - 1, j)
                if j > 0:
                    if dp[i][j - 1] + 1 < dp[i][j]:
                        dp[i][j] = dp[i][j - 1] + 1
                        back[i][j] = ("del", i, j - 1)

        ops = []
        i, j = M, N
        while i > 0 or j > 0:
            op, pi, pj = back[i][j]
            ops.append((op, pi, pj))
            i, j = pi, pj
        ops.reverse()

        # ---- Find the last 'sub' position in ref ----
        last_sub_ref = -1
        for op, pi, pj in ops:
            if op == "sub":
                last_sub_ref = max(last_sub_ref, pi)

        # ---- Build output with INSERT for nearby 'ins' ops ----
        # Rule: INSERT 'ins' ops that are within 1 position of last_sub_ref
        # (they represent chars ASR missed in this segment). 'ins' ops further
        # away are skipped (belong to future segments).
        INSERT_WINDOW = 1  # insert 'ins' within 1 position of last 'sub'

        out_words: list[str] = []
        out_ts: list[list[float]] = []
        consumed_ref = 0

        for op, pi, pj in ops:
            if op == "sub":
                out_words.append(ref_chars[pi])
                out_ts.append(list(asr_ts[pj]))
                consumed_ref = max(consumed_ref, pi + 1)
            elif op == "ins":
                # 'ins' = ref char with no ASR match
                # Only insert if within INSERT_WINDOW of last_sub_ref
                if pi <= last_sub_ref + INSERT_WINDOW:
                    # INSERT: split previous ASR char's timestamp
                    out_words.append(ref_chars[pi])
                    if out_ts:
                        prev_s, prev_e = out_ts[-1]
                        mid = (prev_s + prev_e) / 2.0
                        out_ts[-1] = [prev_s, mid]
                        out_ts.append([mid, prev_e])
                    else:
                        out_ts.append([0.0, 0.0])
                    consumed_ref = max(consumed_ref, pi + 1)
                # else: skip (belongs to future segment)
            elif op == "del":
                # 'del' = extra ASR char with no ref match (melisma)
                # Assign nearest ref char identity, keep ASR timestamp
                out_words.append(ref_chars[min(pi, M - 1)])
                out_ts.append(list(asr_ts[pj]))

        return out_words, out_ts, consumed_ref


def _merge_held_note_sps(
    words: list[str], word_durs: list[float],
) -> tuple[list[str], list[float]]:
    """Merge ``<SP>`` tokens between identical words in the PRE-ROSVOT words list.

    Used when the words list has ``[X, <SP>, X]`` patterns from
    ``_build_words_with_gaps`` (ASR detected a gap inside a held note).
    Merges the SP duration into the first X, so ROSVOT doesn't see the SP.

    Note: this is applied BEFORE ROSVOT. For SPs that ROSVOT itself
    re-introduces, see :func:`_merge_rosvot_held_note_sps`.
    """
    if len(words) < 3:
        return words, word_durs
    out_words: list[str] = []
    out_durs: list[float] = []
    i = 0
    while i < len(words):
        if (out_words and words[i] == "<SP>"
                and i + 1 < len(words)
                and words[i + 1] == out_words[-1]
                and out_words[-1] != "<SP>"):
            out_durs[-1] += word_durs[i]
            out_words.append(words[i + 1])
            out_durs.append(word_durs[i + 1])
            i += 2
        else:
            out_words.append(words[i])
            out_durs.append(word_durs[i])
            i += 1
    return out_words, out_durs


def _merge_rosvot_held_note_sps(metadata: dict) -> None:
    """Fix ``[X, <SP>, X]`` patterns introduced by ROSVOT's note2words.

    ROSVOT sometimes assigns a note to the ``<SP>`` word index even when
    the singer sustained the same syllable across that region (melisma).
    This creates an ``<SP>`` token between two identical char tokens in
    ``note_text``, which causes SoulX-Singer to insert an unnatural pause
    during synthesis.

    This function scans ``note_text`` in-place for the pattern
    ``[X, <SP>, X]`` (where X is any non-SP token) and converts the
    ``<SP>`` to X with:
    - ``note_pitch``: copied from the preceding X (so synthesis produces
      the correct pitch instead of silence)
    - ``note_type``: set to 3 (word-internal continuation)
    - ``phoneme``: copied from the preceding X

    The ``f0`` field is NOT modified — it already contains the singer's
    actual f0 values for that region (which are voiced, not silent).
    """
    nt = metadata.get("note_text", [])
    if not nt or len(nt) < 3:
        return
    np_pitches = metadata.get("note_pitch", [])
    np_types = metadata.get("note_type", [])
    # phoneme is not in the ROSVOT output dict; it's added by convert_metadata.
    # We only fix note_text, note_pitch, note_type here.
    n_fixed = 0
    for i in range(1, len(nt) - 1):
        if (nt[i] == "<SP>"
                and nt[i - 1] != "<SP>"
                and nt[i + 1] == nt[i - 1]):
            # Convert SP to continuation of the held note
            nt[i] = nt[i - 1]
            if i < len(np_pitches):
                np_pitches[i] = np_pitches[i - 1]
            if i < len(np_types):
                np_types[i] = 3
            n_fixed += 1
    if n_fixed:
        print(f"[MIDI-Edit] Fixed {n_fixed} ROSVOT-introduced <SP> in "
              f"held notes for {metadata.get('item_name', '?')}")


def _load_reduplication_wordlist() -> frozenset[str]:
    """Load the valid reduplication (AA-form) word list.

    Reads ``models/reduplication-verbs.txt`` — one AA-form word per line
    (e.g., 哥哥, 滔滔, 纷纷). All entries are 2-char with identical chars.

    Consecutive identical chars whose AA pair is NOT in this list are
    "invalid reduplication" (melisma duplicates) and get merged.
    """
    wordlist_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "reduplication-verbs.txt",
    )
    if not os.path.isfile(wordlist_path):
        return frozenset()
    with open(wordlist_path, "r", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def _merge_invalid_repeated_chars(metadata: dict, wordlist: frozenset[str],
                                   ref_chars: str = "") -> None:
    """Merge consecutive identical chars (melisma duplicates).

    For each run of N identical chars, determine how many to keep:

    1. **With reference lyrics** (``ref_chars`` provided): the max consecutive
       count of that char in the reference. E.g., reference "滔滔" has max
       consecutive run of 2 for 滔 → keep 2. Reference "谁负谁" has max run
       of 1 for 谁 → keep 1.

    2. **Without reference**: if the AA pair is in ``wordlist`` (valid
       reduplication like 哥哥) → keep 2. Otherwise → keep 1.

    Extra chars are merged: their duration is added to the last kept char,
    and the extra note entries are removed. f0 is not modified.
    """
    nt = list(metadata.get("note_text", []))
    if not nt or len(nt) < 2:
        return
    nd = list(metadata.get("note_dur", []))
    np_p = list(metadata.get("note_pitch", []))
    nty = list(metadata.get("note_type", []))

    # Build char→max_consecutive_run from reference (if provided)
    ref_max_run: dict[str, int] = {}
    if ref_chars:
        j = 0
        while j < len(ref_chars):
            ch = ref_chars[j]
            run = 1
            while j + run < len(ref_chars) and ref_chars[j + run] == ch:
                run += 1
            ref_max_run[ch] = max(ref_max_run.get(ch, 0), run)
            j += run

    i = 0
    n_fixed = 0
    while i < len(nt):
        if nt[i] == "<SP>":
            i += 1
            continue
        # Find run of identical chars starting at i
        run_char = nt[i]
        run_end = i + 1
        while run_end < len(nt) and nt[run_end] == run_char:
            run_end += 1
        run_len = run_end - i
        if run_len < 2:
            i = run_end
            continue

        # Determine how many to keep
        if ref_max_run:
            keep = ref_max_run.get(run_char, 1)
        else:
            is_valid_word = (run_char + run_char) in wordlist
            keep = 2 if is_valid_word else 1

        if run_len <= keep:
            i = run_end
            continue

        # Merge extras: add their duration to the last kept char
        last_kept = i + keep - 1
        for k in range(i + keep, run_end):
            if last_kept < len(nd) and k < len(nd):
                nd[last_kept] += nd[k]
        # Remove extra entries (backward to preserve indices)
        for k in range(run_end - 1, i + keep - 1, -1):
            for lst in (nt, nd, np_p, nty):
                if k < len(lst):
                    del lst[k]
        n_fixed += (run_len - keep)
        i += keep

    if n_fixed:
        metadata["note_text"] = nt
        metadata["note_dur"] = nd
        metadata["note_pitch"] = np_p
        metadata["note_type"] = nty
        print(f"[MIDI-Edit] Merged {n_fixed} invalid repeated char(s) in "
              f"{metadata.get('item_name', '?')}")

    if n_fixed:
        metadata["note_text"] = nt
        metadata["note_dur"] = nd
        metadata["note_pitch"] = np_p
        metadata["note_type"] = nty
        print(f"[MIDI-Edit] Merged {n_fixed} invalid repeated char(s) in "
              f"{metadata.get('item_name', '?')}")


def _build_lyrics_text(captured_segment_words: list[list[str]]) -> str:
    """Build a single lyrics-text string from captured per-segment word lists.

    Concatenates every segment's pre-ROSVOT ``words`` list (the same artifact
    that gets fed into ``note_transcriber``), then:

    - Removes inter-token spaces (Chinese chars don't need them; English
      words retain their original boundaries via the leading/trailing space
      pattern in the source ``text`` field).
    - Replaces ``<SP>`` tokens (single or consecutive) with a single newline.

    Output example::

        沧海笑
        滔滔两岸潮
        浮沉随浪记今朝

    This is the *pre-ROSVOT* lyrics: one char per syllable, no melisma
    duplications. Useful for reviewing what ASR / force-alignment produced
    before ROSVOT expands it into the final note-level ``text`` field.
    """
    if not captured_segment_words:
        return ""
    # Concatenate all segments' word lists in order.
    all_words: list[str] = []
    for words in captured_segment_words:
        all_words.extend(words)
    # Join with space, then drop all spaces (Chinese chars don't need them;
    # English words in the source ASR are already single-token).
    text = " ".join(all_words)
    text = text.replace(" ", "")
    # Collapse consecutive <SP> into a single newline.
    text = re.sub(r"(?:<SP>)+", "\n", text)
    return text.strip()


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
                                  max_merge_duration: int = 30000,
                                  reference_lyrics: str | None = None,
                                  merge_held_notes: bool = True,
                                  merge_repeated_chars: bool = True,
                                  ) -> tuple[list[dict], str]:
    """Run SoulX-Singer preprocessing on *audio* and return the raw metadata list.

    When ``reference_lyrics`` is provided, uses a **two-pass** approach:

    1. Run ASR (first + second pass with hotword) on ALL segments, collecting
       per-segment char counts.
    2. Distribute reference chars across segments **proportionally** using the
       largest-remainder method, ensuring ``sum(adjusted_counts) == len(ref)``.
       This handles cases where ASR undercounts (e.g., detects 29 chars but
       reference has 31) — the missing chars are distributed to the segments
       where ASR is most likely to have merged syllables.
    3. Run DTW per segment with the adjusted ref counts. DTW's ``ins`` ops
       **insert** missing ref chars by splitting the adjacent ASR char's
       timestamp in half (so ROSVOT receives correct char-level timing).
    4. Run note_transcriber (ROSVOT) per segment — unchanged.

    This ensures both ``midi_json`` and ``lyrics_text`` contain ALL reference
    chars, even when ASR undercounts.
    """
    with tempfile.TemporaryDirectory(prefix="soulsx_preprocess_") as tmpdir:
        audio_path = _ensure_wav_path(audio, sample_rate, tmpdir)
        save_dir = os.path.join(tmpdir, "output")
        pipeline = _get_preprocess_pipeline(
            language=language, save_dir=save_dir,
            max_merge_duration=max_merge_duration,
        )

        # Always install wrapper for lyrics capture.
        wrapper: _ForceAlignLyricTranscriber | None = None
        if pipeline.lyric_transcriber is not None:
            wrapper = _ForceAlignLyricTranscriber(
                pipeline.lyric_transcriber, reference_lyrics,
            )
            pipeline.lyric_transcriber = wrapper
            if wrapper.reference_chars:
                print(f"[MIDI-Edit] ForceAlign: enabled with "
                      f"{len(wrapper.reference_chars)} reference chars")

        has_ref = (wrapper is not None and wrapper.reference_chars)
        if has_ref:
            metadata_list = _run_two_pass_pipeline(
                pipeline, audio_path, save_dir, language, wrapper,
                max_merge_duration, merge_held_notes, merge_repeated_chars,
            )
        else:
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

    lyrics_text = (
        _build_lyrics_text(wrapper.captured_segment_words)
        if wrapper is not None else ""
    )
    return metadata_list, lyrics_text


def _run_two_pass_pipeline(pipeline, audio_path, save_dir, language,
                            wrapper, max_merge_duration,
                            merge_held_notes=True,
                            merge_repeated_chars=True) -> list[dict]:
    """Reimplement pipeline.run() with two-pass ASR for reference alignment.

    Pass 1: Run ASR on all segments, collect per-segment results.
    Pass 2: Distribute reference proportionally, run DTW + note_transcriber.
    """
    import soundfile as sf
    import librosa
    from pathlib import Path

    # Lazy-load the utilities from the SoulX-Singer submodule.
    from preprocess.tools.lyric_transcription import (
        _build_words_with_gaps, _word_dur_post_process,
    )
    from preprocess.utils import convert_metadata, merge_short_segments

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: vocal separation ----
    sep = pipeline.vocal_separator.process(audio_path)
    vocal = sep.vocals_dereverbed.T
    acc = sep.accompaniment.T
    sample_rate = sep.sample_rate
    vocal_path = output_dir / "vocal.wav"
    sf.write(vocal_path, vocal, sample_rate)

    # ---- Step 2: F0 extraction ----
    vocal_f0 = pipeline.f0_extractor.process(
        str(vocal_path),
        f0_path=str(vocal_path).replace(".wav", "_f0.npy"),
    )

    # ---- Step 3: vocal detection (segmentation) ----
    segments = pipeline.vocal_detector.process(str(vocal_path), f0=vocal_f0)
    print(f"[MIDI-Edit] Two-pass: {len(segments)} segments detected")

    # ---- Pass 1: Run ASR on ALL segments, collect results ----
    # Store per-segment: (seg_dict, asr_words, asr_ts)
    asr_results: list[tuple[dict, list[str], list[list[float]]]] = []
    ref_chars = wrapper.reference_chars
    ref_pos = 0  # temporary position for hotword generation

    for seg in segments:
        wav_fn = seg["wav_fn"]
        # Per-segment f0 (needed by word_dur_post_process)
        pipeline.f0_extractor.process(
            wav_fn, f0_path=wav_fn.replace(".wav", "_f0.npy"),
        )

        # First-pass ASR (no hotword) → rough char count
        out_first = wrapper.inner.zh_model.model.generate(
            wav_fn, output_timestamp=True,
        )[0]
        first_words = out_first["text"].replace("@", "").split(" ")
        n_first = len(first_words)

        # Slice ref for hotword (rough, based on first-pass count)
        hotword_ref = ref_chars[ref_pos:ref_pos + n_first]
        ref_pos += len(hotword_ref)  # advance temp position

        # Second-pass ASR with hotword
        hotword = " ".join(wrapper._split_phrases(hotword_ref)) if hotword_ref else ""
        try:
            out = wrapper.inner.zh_model.model.generate(
                wav_fn, output_timestamp=True, hotword=hotword,
            )[0]
        except Exception:
            out = out_first

        asr_words = out["text"].replace("@", "").split(" ")
        asr_ts = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]
        asr_results.append((seg, asr_words, asr_ts))
        print(f"[MIDI-Edit] Two-pass seg={os.path.basename(wav_fn)}: "
              f"ASR ({len(asr_words)} chars): {' '.join(asr_words)}")

    # ---- Distribute reference chars proportionally ----
    asr_counts = [len(r[1]) for r in asr_results]
    total_asr = sum(asr_counts)
    total_ref = len(ref_chars)
    adjusted_counts = _proportional_distribute(asr_counts, total_ref)
    print(f"[MIDI-Edit] Two-pass: ASR counts={asr_counts} (total={total_asr}), "
          f"ref={total_ref}, adjusted={adjusted_counts}")

    # ---- Pass 2: DTW + note_transcriber per segment ----
    ref_pos = 0  # reset for actual ref consumption
    metadata = []
    wrapper.captured_segment_words = []  # reset capture
    wrapper.diagnostics = []

    for (seg, asr_words, asr_ts), n_ref in zip(asr_results, adjusted_counts):
        wav_fn = seg["wav_fn"]
        # Slice ref by ADJUSTED count (not ASR count)
        seg_ref = ref_chars[ref_pos:ref_pos + n_ref]
        ref_pos += len(seg_ref)

        # DTW with INSERT for 'ins' ops
        if len(seg_ref) > len(asr_words):
            # Ref has MORE chars than ASR — need to INSERT missing chars
            final_words, final_ts = _dtw_align_insert(
                list(seg_ref), asr_words, asr_ts,
            )
            mode = "insert"
        elif len(seg_ref) < len(asr_words):
            # Ref has FEWER chars than ASR — extra ASR chars get nearest ref
            final_words = _ForceAlignLyricTranscriber._dtw_align(
                list(seg_ref), asr_words,
            )
            final_ts = asr_ts
            mode = "collapse"
        else:
            # Equal counts — simple 1:1 DTW
            final_words = _ForceAlignLyricTranscriber._dtw_align(
                list(seg_ref), asr_words,
            )
            final_ts = asr_ts
            mode = "match"

        diffs = [(a, b) for a, b in zip(asr_words, final_words[:len(asr_words)])
                 if a != b]
        print(f"[MIDI-Edit] Two-pass seg={os.path.basename(wav_fn)}: "
              f"DTW ({mode}, {len(final_words)} chars): "
              f"{' '.join(final_words)}"
              + (f"  diffs={diffs}" if diffs else ""))

        wrapper.diagnostics.append({
            "wav_fn": wav_fn, "asr_words": asr_words,
            "final_words": final_words, "diffs": diffs, "ref_slice": seg_ref,
        })

        # Build (words, word_durs)
        words, word_durs = _build_words_with_gaps(final_words, final_ts, wav_fn)
        f0_path = wav_fn.replace(".wav", "_f0.npy")
        if os.path.exists(f0_path):
            words, word_durs = _word_dur_post_process(
                words, word_durs, np.load(f0_path),
            )
        wrapper.captured_segment_words.append(list(words))

        seg["words"] = words
        seg["word_durs"] = word_durs
        seg["language"] = language
        seg_metadata = pipeline.note_transcriber.process(seg, segment_info=seg)
        # Post-ROSVOT fixes:
        # 1. merge_held_notes: fix [X, <SP>, X] patterns (ROSVOT assigned a
        #    note to the <SP> word index inside a sustained syllable)
        if merge_held_notes:
            _merge_rosvot_held_note_sps(seg_metadata)
        # 2. merge_repeated_chars: merge consecutive identical chars that are
        #    NOT valid reduplication words (e.g., 沧沧 from melisma → 沧)
        if merge_repeated_chars:
            wl = _load_reduplication_wordlist()
            if wl:
                _merge_invalid_repeated_chars(seg_metadata, wl, seg_ref)
        metadata.append(seg_metadata)

    # ---- Merge short segments ----
    merged = merge_short_segments(
        vocal, sample_rate, metadata,
        output_dir / "long_cut_wavs",
        max_duration_ms=max_merge_duration,
    )

    # ---- Final F0 + convert_metadata ----
    final_metadata = []
    for item in merged:
        pipeline.f0_extractor.process(
            item.wav_fn, f0_path=item.wav_fn.replace(".wav", "_f0.npy"),
        )
        final_metadata.append(convert_metadata(item))

    # Write metadata.json (so downstream code can read it if needed)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)

    n_fixed = sum(len(d["diffs"]) for d in wrapper.diagnostics)
    if n_fixed:
        print(f"[MIDI-Edit] ForceAlign: corrected {n_fixed} char(s) via DTW")

    return final_metadata


def _proportional_distribute(counts: list[int], target: int) -> list[int]:
    """Distribute ``target`` across buckets proportional to ``counts``.

    Uses the largest-remainder method to ensure ``sum(result) == target``.
    """
    if not counts or target <= 0:
        return [0] * len(counts)
    total = sum(counts)
    if total == 0:
        # Even distribution
        base = target // len(counts)
        rem = target % len(counts)
        return [base + (1 if i < rem else 0) for i in range(len(counts))]

    # Exact proportions
    exact = [c * target / total for c in counts]
    # Floor
    floor = [int(e) for e in exact]
    deficit = target - sum(floor)
    if deficit <= 0:
        return floor

    # Distribute deficit to largest remainders
    remainders = [(exact[i] - floor[i], i) for i in range(len(counts))]
    remainders.sort(reverse=True)
    for k in range(deficit):
        floor[remainders[k % len(counts)][1]] += 1
    return floor


def _dtw_align_insert(
    ref_chars: list[str],
    asr_words: list[str],
    asr_ts: list[list[float]],
) -> tuple[list[str], list[list[float]]]:
    """DTW that INSERTS missing ref chars by splitting ASR timestamps.

    When ``len(ref) > len(asr)`` (ASR undercounted), the missing ref chars
    are inserted into the output by splitting the adjacent ASR char's
    timestamp. This gives ROSVOT correct char-level timing for every
    reference char.

    Returns ``(out_words, out_ts)`` where ``len(out_words) == len(ref_chars)``.
    """
    M, N = len(ref_chars), len(asr_words)
    if N == 0:
        return [], []
    if M == 0:
        return list(asr_words), asr_ts
    if M == N:
        # Equal: simple 1:1 DTW (replace identity only)
        words = _ForceAlignLyricTranscriber._dtw_align(ref_chars, asr_words)
        return words, [list(ts) for ts in asr_ts]

    # M > N: need to insert M-N chars. Use DTW to find where.
    INF = float("inf")
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    back = [[None] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and j > 0:
                cost = 0 if ref_chars[i - 1] == asr_words[j - 1] else 1
                if dp[i - 1][j - 1] + cost < dp[i][j]:
                    dp[i][j] = dp[i - 1][j - 1] + cost
                    back[i][j] = ("sub", i - 1, j - 1)
            if i > 0:
                if dp[i - 1][j] + 1 < dp[i][j]:
                    dp[i][j] = dp[i - 1][j] + 1
                    back[i][j] = ("ins", i - 1, j)
            if j > 0:
                if dp[i][j - 1] + 1 < dp[i][j]:
                    dp[i][j] = dp[i][j - 1] + 1
                    back[i][j] = ("del", i, j - 1)

    # Trace back
    ops = []
    i, j = M, N
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]
        ops.append((op, pi, pj))
        i, j = pi, pj
    ops.reverse()

    # Build output: every ref char gets an output entry. 'ins' chars
    # borrow timing from the PREVIOUS 'sub' char (split in half).
    out_words: list[str] = []
    out_ts: list[list[float]] = []

    for op, pi, pj in ops:
        if op == "sub":
            out_words.append(ref_chars[pi])
            out_ts.append(list(asr_ts[pj]))
        elif op == "ins":
            # Insert ref char, split previous ASR char's timestamp
            out_words.append(ref_chars[pi])
            if out_ts:
                prev_s, prev_e = out_ts[-1]
                mid = (prev_s + prev_e) / 2.0
                out_ts[-1] = [prev_s, mid]
                out_ts.append([mid, prev_e])
            else:
                # No previous — borrow from next (will be fixed by _build_words_with_gaps)
                out_ts.append([0.0, 0.0])
        elif op == "del":
            # This shouldn't happen when M > N, but handle gracefully
            out_words.append(ref_chars[min(pi, M - 1)])
            out_ts.append(list(asr_ts[pj]))

    return out_words, out_ts


def transcribe_audio(audio, sample_rate: int | None = None,
                     language: str = "Mandarin",
                     max_merge_duration: int = 30000,
                     reference_lyrics: str | None = None,
                     return_lyrics: bool = False,
                     merge_held_notes: bool = True,
                     merge_repeated_chars: bool = True):
    """Transcribe audio to MIDI JSON using SoulX-Singer preprocessing.

    Args:
        audio: File path (str/Path), numpy array, or (array, sample_rate) tuple.
        sample_rate: Required only when audio is a bare numpy array.
        language: Lyric language — "Mandarin", "English", or "Cantonese".
        max_merge_duration: Max segment merge duration in ms.
        reference_lyrics: Optional reference lyrics to force text accuracy to
            100%. When provided, ASR decoding is biased (via Paraformer
            ``hotword``) and post-aligned (via char-level DTW) so the output
            text exactly matches the reference. SPs / pitch / duration / f0 /
            melisma structure are still driven by the audio — only the *text*
            identity of each token is replaced. Punctuation/whitespace in the
            reference are stripped (they only serve as semantic slicing hints
            for the user's own reading; SP placement is audio-driven).
        return_lyrics: When True, returns ``(midi_json, lyrics_text)`` tuple
            where ``lyrics_text`` is the pre-ROSVOT lyrics (one char per
            syllable, no melisma duplications) with ``<SP>`` tokens replaced by
            newlines. Default False for backward compatibility (returns MIDI
            JSON string only).

    Returns:
        MIDI JSON string by default, or ``(midi_json, lyrics_text)`` tuple
        when ``return_lyrics=True``.
    """
    metadata_list, lyrics_text = _preprocess_audio_to_metadata(
        audio, sample_rate=sample_rate, language=language,
        max_merge_duration=max_merge_duration,
        reference_lyrics=reference_lyrics,
        merge_held_notes=merge_held_notes,
        merge_repeated_chars=merge_repeated_chars,
    )
    midi_json = metadata_to_midi_json(metadata_list)
    if return_lyrics:
        return midi_json, lyrics_text
    return midi_json


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
    # _preprocess_audio_to_metadata returns (metadata, lyrics_text); we only
    # need the metadata for prompt purposes.
    return _preprocess_audio_to_metadata(
        fallback_audio,
        sample_rate=fallback_sample_rate,
        language=fallback_language,
    )[0]


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
