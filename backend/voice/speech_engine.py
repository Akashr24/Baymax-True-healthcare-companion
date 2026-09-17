"""
Baymax Empathic Speech Synthesis Engine.
Provides pitch, pace, and warmth modulation matching Baymax's calm bedside manner.
Supports both local pyttsx3 audio generation and browser Web Speech parameters.
"""

import os
import tempfile
from typing import Any, Dict, Optional


class BaymaxSpeechEngine:
    """
    Affective Speech Engine for Baymax.
    Dynamically modulates vocal pace, pitch, and timbre based on patient stress,
    pain, and clinical triage urgency.
    """

    def __init__(self):
        self._init_tts()

    def _init_tts(self):
        """Initializes pyttsx3 offline engine safely."""
        self.engine = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # Set default gentle pace
            self.engine.setProperty("rate", 140)
            self.engine.setProperty("volume", 0.95)
        except Exception:
            self.engine = None

    def calculate_voice_modulation(
        self,
        stress_level: int = 20,
        pain_score: int = 0,
        esi_level: int = 4,
    ) -> Dict[str, Any]:
        """
        Calculates optimal speech synthesis parameters (for Web Speech API and backend).
        """
        # If patient is in severe distress, Baymax speaks slower and more soothingly
        if stress_level > 65 or pain_score >= 8:
            # Calming mode
            rate = 0.82  # Slower pace for grounding
            pitch = 0.88  # Softer, lower frequency to reduce arousal
            volume = 0.95
            persona_mode = "Calming / Grounding"
        elif esi_level in [1, 2]:
            # Emergency escalation mode
            rate = 0.95  # Clear, steady
            pitch = 0.96
            volume = 1.0
            persona_mode = "Direct Emergency Protocol"
        elif pain_score >= 5:
            # Gentle comforting mode
            rate = 0.88
            pitch = 0.90
            volume = 0.90
            persona_mode = "Comforting / Gentle"
        else:
            # Normal Baymax baseline
            rate = 0.90
            pitch = 0.92
            volume = 0.95
            persona_mode = "Baymax Standard Healthcare"

        return {
            "rate": round(rate, 2),
            "pitch": round(pitch, 2),
            "volume": round(volume, 2),
            "persona_mode": persona_mode,
        }

    def synthesize_to_file(
        self,
        text: str,
        stress_level: int = 20,
        pain_score: int = 0,
        esi_level: int = 4
    ) -> Optional[str]:
        """
        Synthesizes speech to a temporary WAV/MP3 file if pyttsx3 is functional.
        Returns path to audio file.
        """
        if not self.engine:
            return None

        modulation = self.calculate_voice_modulation(stress_level, pain_score, esi_level)
        # Convert 0.8-1.0 rate to pyttsx3 words-per-minute (standard 140)
        wpm = int(modulation["rate"] * 150.0)

        try:
            self.engine.setProperty("rate", wpm)
            temp_dir = tempfile.gettempdir()
            out_file = os.path.join(temp_dir, f"baymax_speech_{os.getpid()}.wav")
            self.engine.save_to_file(text, out_file)
            self.engine.runAndWait()
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                return out_file
        except Exception:
            return None
        return None
