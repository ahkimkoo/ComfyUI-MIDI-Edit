try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Relative import resolves under ComfyUI's loader (which sets __package__).
    # When imported as a top-level module (e.g. by pytest collecting the repo
    # root as a Package), the relative import has no parent package — fall back
    # to empty mappings so the test harness can run. No-op in production.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
