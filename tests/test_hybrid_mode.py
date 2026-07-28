# tests/test_hybrid_mode.py
"""Unit tests for the experimental ``control="hybrid"`` SVS mode.

Background
----------
SoulX-Singer's ``SoulXSinger.infer`` supports two control modes:

* ``score``  — uses discrete ``note_pitch`` (clear diction, but the intra-note
  f0 trajectory is lost → pitch can drift).
* ``melody`` — uses the continuous ``f0`` contour (correct pitch trajectory,
  but diction can blur).

The model architecture *adds* the ``note_pitch_encoder`` and ``f0_encoder``
outputs (``soulxsinger/models/soulxsinger.py:179-183``), so feeding BOTH at
once is architecturally supported. The hybrid mode does exactly that: it keeps
both ``note_pitch`` and ``f0`` non-None, instead of zeroing one of them.

We don't touch the ``SoulX-Singer`` submodule; instead we monkey-patch
``model.infer`` from our own ``core/soulsx_singer.py`` via
``_enable_hybrid_mode`` / ``_disable_hybrid_mode``.

The test suite follows the assignment's four required cases:
  1. enable/disable lifecycle (patch applied / restored),
  2. non-hybrid controls delegate to the original infer,
  3. hybrid control supplies BOTH note_pitch and f0 to the model,
  4. the ComfyUI node exposes 'hybrid' in its control options.

Plus graceful-degradation and auto_shift coverage carried over from prior work.
"""
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure the workspace dir is on sys.path so `from core.soulsx_singer import ...`
# resolves to THIS package (pytest's rootdir is the ComfyUI root, which can
# shadow local modules otherwise).
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from core.soulsx_singer import (  # noqa: E402
    _disable_hybrid_mode,
    _enable_hybrid_mode,
    _extract_hybrid_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_f0_to_coarse(f0, f0_bin=361, f0_min=32.7031956625, f0_shift=0):
    """Faithful copy of upstream ``SoulXSinger.f0_to_coarse`` (CPU/torch path).

    Importing the real ``SoulXSinger`` pulls heavy deps (``accelerate`` etc.)
    that aren't installed in the minimal test env, so we replicate the
    binning logic here. Behaviour matches upstream for the torch path.
    """
    is_torch = isinstance(f0, torch.Tensor)
    uv_mask = f0 <= 0
    f0_safe = torch.maximum(f0, torch.tensor(f0_min))
    f0_cents = 1200 * torch.log2(f0_safe / f0_min)
    f0_coarse = (f0_cents / 20) + 1
    f0_coarse = torch.round(f0_coarse).long()
    f0_coarse = torch.clamp(f0_coarse, min=1, max=f0_bin - 1)
    f0_coarse[uv_mask] = 0
    if f0_shift != 0:
        voiced = f0_coarse > 0
        if voiced.any():
            shifted = f0_coarse[voiced] + f0_shift
            f0_coarse[voiced] = torch.clamp(shifted, 1, f0_bin - 1)
    return f0_coarse


def _build_mock_model(B=1, T=4, F=8, H=2):
    """A mock SoulXSinger with a real ``f0_to_coarse`` and stubbed NN parts.

    NN parts return real tensors of a fixed shape so the torch ops (cat, add,
    slicing) inside the hybrid infer body succeed; we only inspect *which*
    signals reach the encoders, not the audio output.

    ``_hybrid_patched`` / ``_hybrid_requested`` are pre-set to False so the
    install guard behaves like a real ``nn.Module`` (which raises
    ``AttributeError`` for unknown attrs) instead of MagicMock's
    auto-creates-a-truthy-child behaviour.
    """
    model = MagicMock()
    model._hybrid_patched = False
    model._hybrid_requested = False
    model.f0_to_coarse = _stub_f0_to_coarse
    feat_t = torch.zeros(B, T, H)
    feat_f = torch.zeros(B, F, H)
    model.note_pitch_encoder = MagicMock(return_value=feat_t)
    model.note_type_encoder = MagicMock(return_value=feat_t)
    model.note_text_encoder = MagicMock(return_value=feat_t)
    model.preflow = MagicMock(return_value=feat_t)
    model.expand_states = MagicMock(return_value=feat_f)
    model.f0_encoder = MagicMock(return_value=feat_f)
    model.mel = MagicMock(return_value=feat_f)
    model.cfm_decoder = MagicMock()
    model.cfm_decoder.reverse_diffusion = MagicMock(return_value=feat_f)
    model.vocoder = MagicMock(return_value=torch.zeros(1, 1, 8))
    return model


def _build_meta_both_signals():
    """meta dict carrying BOTH non-zero note_pitch and non-zero f0."""
    B = 1
    return {
        "target": {
            "phoneme": torch.randint(1, 10, (B, 2)).long(),
            "mel2note": torch.zeros(B, 5).long(),
            "note_type": torch.ones(B, 2).long(),
            "note_pitch": torch.tensor([[60, 62]]).long(),
            "f0": torch.tensor([[440.0, 0.0, 440.0, 0.0, 440.0]]).float(),
        },
        "prompt": {
            "waveform": torch.zeros(B, 16000).float(),
            "phoneme": torch.randint(1, 10, (B, 2)).long(),
            "mel2note": torch.zeros(B, 3).long(),
            "note_type": torch.ones(B, 2).long(),
            "note_pitch": torch.tensor([[55, 0]]).long(),
            "f0": torch.tensor([[220.0, 220.0, 0.0]]).float(),
        },
    }


# ---------------------------------------------------------------------------
# 1. Patch lifecycle: _enable_hybrid_mode / _disable_hybrid_mode
# ---------------------------------------------------------------------------


class TestPatchLifecycle:
    """Enable replaces model.infer; disable restores it; both are idempotent."""

    def test_enable_disable_hybrid_mode(self):
        model = _build_mock_model()
        original_infer = model.infer

        _enable_hybrid_mode(model)

        # Patch applied: infer replaced, flag set, original saved.
        assert model._hybrid_patched is True
        assert model.infer is not original_infer
        assert model._orig_infer is original_infer

        _disable_hybrid_mode(model)

        # Patch removed: infer restored, flags cleared.
        assert model._hybrid_patched is False
        assert model._hybrid_requested is False
        assert model.infer is original_infer
        assert not hasattr(model, "_orig_infer")

    def test_enable_is_idempotent(self):
        """Calling _enable_hybrid_mode twice must not double-wrap or lose the original."""
        model = _build_mock_model()
        original_infer = model.infer

        _enable_hybrid_mode(model)
        first_patch = model.infer
        _enable_hybrid_mode(model)  # second call: no-op

        assert model.infer is first_patch
        assert model._orig_infer is original_infer

        _disable_hybrid_mode(model)
        assert model.infer is original_infer

    def test_disable_without_enable_is_safe(self):
        """Calling _disable_hybrid_mode on an un-patched model is a no-op."""
        model = _build_mock_model()
        original_infer = model.infer
        _disable_hybrid_mode(model)  # must not raise
        assert model.infer is original_infer


# ---------------------------------------------------------------------------
# 2. Non-hybrid controls delegate to the original infer
# ---------------------------------------------------------------------------


class TestDelegation:
    """control='score'/'melody' must be forwarded to the original infer verbatim."""

    @pytest.mark.parametrize("mode", ["score", "melody"])
    def test_hybrid_infer_delegates_non_hybrid(self, mode):
        model = _build_mock_model()
        original = model.infer  # MagicMock child attribute — the "original" infer
        _enable_hybrid_mode(model)

        meta = {"target": {}, "prompt": {}}
        model.infer(meta, auto_shift=False, pitch_shift=0, control=mode, use_fp16=False)

        assert original.called, "original infer must handle non-hybrid control"
        assert original.call_args.kwargs.get("control") == mode


# ---------------------------------------------------------------------------
# 3. Hybrid control supplies BOTH note_pitch and f0 to the model
# ---------------------------------------------------------------------------


class TestHybridBothSignals:
    """control='hybrid' must drive BOTH encoders with non-zero input."""

    def test_hybrid_infer_provides_both_signals(self):
        model = _build_mock_model()
        original = model.infer
        _enable_hybrid_mode(model)

        meta = _build_meta_both_signals()
        model.infer(meta, auto_shift=False, pitch_shift=0, control="hybrid")

        # Hybrid must run the patched body, not the original infer (which would
        # raise ValueError on control="hybrid").
        assert not original.called, "hybrid must run the patched body, not original infer"

        # note_pitch_encoder received the concatenated REAL note pitches (non-zero).
        captured_np = model.note_pitch_encoder.call_args.args[0].flatten().tolist()
        assert {55, 60, 62}.issubset(set(captured_np)), (
            f"note_pitch_encoder did not receive real pitches: {captured_np}"
        )
        # f0 path active (f0 not zeroed out before binning).
        assert model.f0_encoder.called, "f0_encoder must be called in hybrid mode"

    def test_hybrid_triggered_by_override_flag(self):
        """synthesize_audio passes args.control='score' (to bypass process
        validation) plus model._hybrid_requested=True to signal hybrid intent.
        The patched infer must do hybrid when the flag is set, even though
        control='score' on paper."""
        model = _build_mock_model()
        original = model.infer
        _enable_hybrid_mode(model)
        model._hybrid_requested = True  # set by synthesize_audio when control="hybrid"

        meta = _build_meta_both_signals()
        # control='score' on paper, but _hybrid_requested forces hybrid path.
        model.infer(meta, auto_shift=False, pitch_shift=0, control="score")

        assert not original.called, "_hybrid_requested=True must trigger hybrid body"
        assert model.note_pitch_encoder.called
        assert model.f0_encoder.called

    def test_hybrid_auto_shift_uses_note_pitch_median(self):
        """auto_shift in hybrid mode must use the note_pitch median (like score).

        When both note_pitch and f0 are present, upstream's auto_shift prefers
        note_pitch (the first branch). The hybrid body copies that branch
        verbatim, so auto_shift must run without error and drive the pipeline
        to completion.
        """
        model = _build_mock_model()
        _enable_hybrid_mode(model)

        # Target an octave below prompt (note_pitch diff = +12 when shifting up).
        meta = _build_meta_both_signals()
        meta["target"]["note_pitch"] = torch.full((1, 2), 48, dtype=torch.long)
        meta["prompt"]["note_pitch"] = torch.full((1, 2), 60, dtype=torch.long)

        result = model.infer(meta, auto_shift=True, pitch_shift=0, control="hybrid")
        assert result is not None
        assert model.note_pitch_encoder.called


# ---------------------------------------------------------------------------
# 3b. Graceful degradation when a signal is missing
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Missing note_pitch or f0 degrades cleanly (zeros substituted), never raises."""

    def test_extract_returns_none_for_missing_f0(self):
        meta = {
            "target": {"note_pitch": [60]},
            "prompt": {"note_pitch": [55]},
        }
        gnp, pnp, gf0, pf0 = _extract_hybrid_signals(meta)
        assert gnp == [60] and pnp == [55]
        assert gf0 is None and pf0 is None

    def test_hybrid_infer_when_f0_missing(self):
        model = _build_mock_model()
        _enable_hybrid_mode(model)

        meta = _build_meta_both_signals()
        del meta["target"]["f0"]
        del meta["prompt"]["f0"]

        result = model.infer(meta, auto_shift=False, pitch_shift=0, control="hybrid")
        assert result is not None
        assert model.f0_encoder.called, "f0_encoder still called (on zero-filled f0)"

    def test_hybrid_infer_when_note_pitch_missing(self):
        model = _build_mock_model()
        _enable_hybrid_mode(model)

        meta = _build_meta_both_signals()
        del meta["target"]["note_pitch"]
        del meta["prompt"]["note_pitch"]

        result = model.infer(meta, auto_shift=False, pitch_shift=0, control="hybrid")
        assert result is not None
        assert model.note_pitch_encoder.called


# ---------------------------------------------------------------------------
# 4. Node surface: MIDISynthesizeAudio accepts "hybrid"
# ---------------------------------------------------------------------------


# TestNodeSurface removed: hybrid mode is no longer exposed in the node UI
# (user feedback: sounds identical to melody). Implementation code is kept
# in core/soulsx_singer.py for potential future use; the tests above still
# cover the monkey-patch logic.
