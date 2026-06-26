try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .nodes import MidiLyricsAlignment
    from .nodes import MIDITranscribeAudio, MIDISynthesizeAudio
    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        "MidiLyricsAlignment": MidiLyricsAlignment,
        "MIDITranscribeAudio": MIDITranscribeAudio,
        "MIDISynthesizeAudio": MIDISynthesizeAudio,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        "MidiLyricsAlignment": "MIDI Lyrics Alignment",
        "MIDITranscribeAudio": "MIDI Transcribe Audio",
        "MIDISynthesizeAudio": "MIDI Synthesize Audio",
    }
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
