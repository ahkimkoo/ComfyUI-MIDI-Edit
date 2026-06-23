try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    # Task 10: register the unified DP-based alignment node. The class itself
    # lives at the bottom of nodes.py; we wire it into the mappings here so
    # nodes.py only gains a new class (no churn to the existing mappings dict).
    from .nodes import MidiLyricsAlignment
    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        "MidiLyricsAlignment": MidiLyricsAlignment,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        "MidiLyricsAlignment": "MIDI Lyrics Alignment (DP)",
    }
except ImportError:
    # Relative import resolves under ComfyUI's loader (which sets __package__).
    # When imported as a top-level module (e.g. by pytest collecting the repo
    # root as a Package), the relative import has no parent package — fall back
    # to empty mappings so the test harness can run. No-op in production.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
