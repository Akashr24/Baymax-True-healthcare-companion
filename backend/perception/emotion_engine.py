"""
Affective Emotion & Acoustic Tone Analyzer.
Analyzes patient voice prosody, pitch jitter, shimmer, and speaking pacing to detect
stress, distress, pain, and panic levels.
"""

from typing import Any, Dict
import numpy as np


class EmotionToneEngine:
    """
    Extracts vocal affective markers from audio input to quantify patient
    emotional distress, stress levels, and pain indicators in real time.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze_vocal_tone(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculates pitch (F0), jitter, shimmer, and vocal energy to estimate
        stress, pain, and panic.
        """
        if audio_data.dtype == np.int16:
            audio_f = audio_data.astype(np.float32) / 32768.0
        else:
            audio_f = audio_data.astype(np.float32)

        if audio_f.ndim > 1:
            audio_f = np.mean(audio_f, axis=1)

        rms = float(np.sqrt(np.mean(audio_f**2) + 1e-9))

        # Check for sufficient vocalization amplitude
        if rms < 0.02 or len(audio_f) < self.sample_rate * 0.3:
            return {
                "stress_level": 15,
                "pain_indicator": 1,
                "panic_index": 0.05,
                "emotional_tone": "Calm & Stable",
                "pitch_f0": 130.0,
                "is_vocalizing": False,
            }

        # 1. Pitch (F0) estimation via Normalized Autocorrelation
        min_lag = int(self.sample_rate / 450.0)  # Max 450 Hz
        max_lag = int(self.sample_rate / 65.0)  # Min 65 Hz

        corr = np.correlate(audio_f, audio_f, mode="full")
        corr = corr[len(corr) // 2 :]

        if len(corr) > max_lag:
            search_window = corr[min_lag:max_lag]
            peak_lag = min_lag + int(np.argmax(search_window))
            f0 = (
                float(self.sample_rate / peak_lag)
                if peak_lag > 0
                else 150.0
            )
        else:
            f0 = 150.0

        # 2. Local Jitter (period-to-period frequency fluctuations)
        # Higher in acute stress, crying, trembling, or pain
        frame_len = int(self.sample_rate * 0.04)  # 40ms subframes
        sub_f0s = []
        sub_amplitudes = []

        for start in range(0, len(audio_f) - frame_len, frame_len):
            chunk = audio_f[start : start + frame_len]
            sub_rms = np.sqrt(np.mean(chunk**2))
            sub_amplitudes.append(sub_rms)

            c = np.correlate(chunk, chunk, mode="full")
            c = c[len(c) // 2 :]
            if len(c) > max_lag:
                p_lag = min_lag + int(np.argmax(c[min_lag:max_lag]))
                if p_lag > 0:
                    sub_f0s.append(self.sample_rate / p_lag)

        jitter = 0.02
        if len(sub_f0s) >= 3:
            diffs = np.abs(np.diff(sub_f0s))
            mean_f0 = np.mean(sub_f0s) + 1e-6
            jitter = float(np.mean(diffs) / mean_f0)

        # 3. Shimmer (amplitude variations between cycles)
        shimmer = 0.05
        if len(sub_amplitudes) >= 3:
            amp_diffs = np.abs(np.diff(sub_amplitudes))
            mean_amp = np.mean(sub_amplitudes) + 1e-6
            shimmer = float(np.mean(amp_diffs) / mean_amp)

        # 4. Affective Metric Calculations
        # High F0 (> 250 Hz in adults), high jitter (> 0.08), high shimmer (> 0.15) indicate distress
        stress_score = 20.0
        stress_score += min(35.0, max(0.0, (f0 - 160.0) * 0.25))
        stress_score += min(25.0, jitter * 250.0)
        stress_score += min(20.0, shimmer * 70.0)
        stress_score = int(np.clip(stress_score, 10, 98))

        # Panic index
        panic_index = float(
            np.clip((stress_score - 30) / 65.0 + (jitter * 2.0), 0.0, 1.0)
        )

        # Pain vocal indicator (1-10)
        pain_indicator = int(np.clip(1 + (stress_score / 11.0), 1, 10))

        # Tone categorization
        if panic_index > 0.75:
            emotional_tone = "Panic / Acute Agony"
        elif stress_score > 65:
            emotional_tone = "Severe Emotional Distress / Pain"
        elif stress_score > 40:
            emotional_tone = "Moderate Anxiety / Strain"
        else:
            emotional_tone = "Calm & Stable"

        return {
            "stress_level": stress_score,
            "pain_indicator": pain_indicator,
            "panic_index": round(panic_index, 2),
            "emotional_tone": emotional_tone,
            "pitch_f0": round(f0, 1),
            "jitter": round(jitter, 3),
            "shimmer": round(shimmer, 3),
            "is_vocalizing": True,
        }
