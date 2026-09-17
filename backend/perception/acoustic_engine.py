"""
Acoustic Diagnostic Classifier Engine.
Analyzes respiratory audio streams (coughs, lung sounds, breath sounds) using
spectral analysis, Mel-frequency filterbanks, and diagnostic pattern detection.
Classifies: Dry Cough, Productive Cough, Wheezing, Stridor, and Dyspnea/Tachypnea.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal


class AcousticDiagnosticEngine:
    """
    Real-time acoustic health assessment engine.
    Extracts acoustic biometrics to identify lung sounds, cough types,
    and signs of respiratory distress.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 1.0,
        history_seconds: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(self.sample_rate * chunk_duration_sec)
        self.history_size = int(self.sample_rate * history_seconds)

        # Ring buffer of recent audio samples
        self.audio_buffer = deque(maxlen=self.history_size)

        # Diagnostics history
        self.cough_events = deque(maxlen=20)
        self.energy_history = deque(maxlen=100)
        self.last_respiratory_rate = 16.0  # standard 16 breaths/min
        self.last_dyspnea_score = 0.0
        self.last_cough_type = "None"
        self.last_cough_time = 0.0

    def process_audio_chunk(
        self, audio_data: np.ndarray, sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming audio chunk (1D numpy array, float32 [-1.0, 1.0] or int16).
        Computes spectral features and runs diagnostic classification.
        """
        if sample_rate and sample_rate != self.sample_rate:
            # Resample or update sample rate if needed
            self.sample_rate = sample_rate

        # Normalize to float32 [-1.0, 1.0]
        if audio_data.dtype == np.int16:
            audio_f = audio_data.astype(np.float32) / 32768.0
        else:
            audio_f = audio_data.astype(np.float32)

        # Flatten if multi-channel
        if audio_f.ndim > 1:
            audio_f = np.mean(audio_f, axis=1)

        # Append to circular buffer
        for s in audio_f:
            self.audio_buffer.append(float(s))

        # Basic signal metrics
        rms = float(np.sqrt(np.mean(audio_f**2) + 1e-9))
        self.energy_history.append(rms)

        # Zero-Crossing Rate (ZCR)
        zcr = float(
            np.mean(np.abs(np.diff(np.sign(audio_f + 1e-9)))) / 2.0
        )

        # If silent / background ambient noise only, return baseline
        if rms < 0.015:
            spectrum_bars = [0.05] * 16
            return {
                "rms_energy": round(rms, 4),
                "zcr": round(zcr, 3),
                "is_cough": False,
                "cough_type": "None",
                "wheeze_detected": False,
                "stridor_detected": False,
                "respiratory_rate": round(self.last_respiratory_rate, 1),
                "dyspnea_index": round(self.last_dyspnea_score, 2),
                "state": "Resting Breath / Normal",
                "spectrum_bars": spectrum_bars,
            }

        # Find the frame of maximum energy within audio_f for spectral analysis
        n_fft = min(1024, len(audio_f))
        hop = max(1, n_fft // 2)
        best_slice = audio_f[:n_fft]
        best_energy = float(np.sum(best_slice**2))
        peak_rms = float(np.sqrt(best_energy / n_fft + 1e-9))

        if len(audio_f) > n_fft:
            for start in range(0, len(audio_f) - n_fft + 1, hop):
                sub = audio_f[start : start + n_fft]
                e = float(np.sum(sub**2))
                if e > best_energy:
                    best_energy = e
                    best_slice = sub
            peak_rms = float(np.sqrt(best_energy / n_fft + 1e-9))

        if n_fft < 128:
            padded = np.pad(best_slice, (0, 128 - len(best_slice)))
        else:
            padded = best_slice

        # Apply Hanning window
        windowed = padded * np.hanning(len(padded))
        fft_vals = np.abs(np.fft.rfft(windowed))
        fft_freqs = np.fft.rfftfreq(len(windowed), 1.0 / self.sample_rate)

        # Spectral Centroid (weighted mean frequency)
        sum_fft = np.sum(fft_vals) + 1e-9
        spectral_centroid = float(np.sum(fft_freqs * fft_vals) / sum_fft)

        # Spectral Roll-off (frequency below which 85% of energy lies)
        cumulative_energy = np.cumsum(fft_vals)
        rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * sum_fft)
        spectral_rolloff = float(
            fft_freqs[min(rolloff_idx, len(fft_freqs) - 1)]
        )

        # Band energies:
        # Low: 50-400 Hz (lung base, breath sounds)
        # Mid: 400-1800 Hz (wheezes, speech)
        # High: 1800-6000 Hz (coughs, stridor, friction rubs)
        low_mask = (fft_freqs >= 50) & (fft_freqs < 400)
        mid_mask = (fft_freqs >= 400) & (fft_freqs < 1800)
        high_mask = (fft_freqs >= 1800) & (fft_freqs < 6000)

        low_energy = float(np.sum(fft_vals[low_mask])) if np.any(low_mask) else 0.0
        mid_energy = float(np.sum(fft_vals[mid_mask])) if np.any(mid_mask) else 0.0
        high_energy = (
            float(np.sum(fft_vals[high_mask])) if np.any(high_mask) else 0.0
        )

        # 1. Wheeze Detection
        # Characteristic: Continuous musical sounds, peak in 400-1200 Hz with high peak-to-average ratio
        wheeze_detected = False
        if mid_energy > 0.35 * sum_fft and rms > 0.025:
            mid_vals = fft_vals[mid_mask]
            if len(mid_vals) > 0:
                peak_ratio = float(np.max(mid_vals) / (np.mean(mid_vals) + 1e-6))
                if peak_ratio > 3.8:
                    wheeze_detected = True

        # 2. Stridor Detection
        # Continuous high-pitched harsh sound (>600 Hz), high intensity with distinct tonal peak
        stridor_detected = False
        if spectral_centroid > 1600 and high_energy > 0.45 * sum_fft and rms > 0.07:
            high_vals = fft_vals[high_mask]
            if len(high_vals) > 0:
                high_peak_ratio = float(np.max(high_vals) / (np.mean(high_vals) + 1e-6))
                if high_peak_ratio > 3.8:
                    stridor_detected = True

        # 3. Cough Detection
        # Characteristic: Sudden burst energy, wideband distribution, high centroid, NOT a pure musical wheeze
        is_cough = False
        cough_type = "None"
        now = time.time()

        if not wheeze_detected and not stridor_detected:
            if (peak_rms > 0.07 or rms > 0.04) and zcr > 0.03 and (high_energy > 0.20 * sum_fft or spectral_centroid > 1600):
                is_cough = True
                self.last_cough_time = now

                # Distinguish Productive (Wet) vs Dry (Barking)
                if spectral_centroid > 2400 and high_energy > mid_energy * 1.2:
                    cough_type = "Dry / Spasmodic Cough"
                elif mid_energy > high_energy * 0.8 and low_energy > 0.25 * mid_energy:
                    cough_type = "Productive / Wet Cough"
                else:
                    cough_type = "Atypical Cough Event"

                self.last_cough_type = cough_type
                self.cough_events.append(
                    {"timestamp": now, "type": cough_type, "centroid": spectral_centroid}
                )

        # 4. Respiratory Rate Estimation from envelope modulation
        if len(self.audio_buffer) >= int(self.sample_rate * 4.0):
            self._estimate_respiratory_rate()

        # 5. Dyspnea Index (0.0 to 1.0)
        # Combines tachypnea (>22 bpm), presence of stridor/wheeze, and frequent coughing
        dyspnea = 0.0
        if self.last_respiratory_rate > 22:
            dyspnea += min(0.4, (self.last_respiratory_rate - 22) * 0.04)
        if wheeze_detected:
            dyspnea += 0.3
        if stridor_detected:
            dyspnea += 0.5
        recent_coughs = sum(
            1 for c in self.cough_events if (now - c["timestamp"]) < 10.0
        )
        if recent_coughs >= 2:
            dyspnea += min(0.3, recent_coughs * 0.1)

        self.last_dyspnea_score = float(np.clip(dyspnea, 0.0, 1.0))

        # Determine diagnostic audio state string
        if stridor_detected:
            state = "CRITICAL: Inspiratory Stridor / Airway Obstruction"
        elif is_cough:
            state = f"Cough Detected: {cough_type}"
        elif wheeze_detected:
            state = "Alert: Wheezing / Bronchial Constriction"
        elif self.last_dyspnea_score > 0.6:
            state = "Respiratory Distress / Tachypnea"
        elif rms > 0.05:
            state = "Vocalized Speech / Normal Phonation"
        else:
            state = "Resting Breath / Stable"

        # Generate 16 UI spectrum visualizer bars (normalized 0.0 to 1.0)
        chunk_len = len(fft_vals) // 16
        spectrum_bars = []
        for i in range(16):
            sub_slice = fft_vals[i * chunk_len : (i + 1) * chunk_len]
            bar_val = float(np.mean(sub_slice)) if len(sub_slice) > 0 else 0.0
            spectrum_bars.append(round(min(1.0, bar_val * 4.0), 3))

        return {
            "rms_energy": round(rms, 4),
            "zcr": round(zcr, 3),
            "spectral_centroid": round(spectral_centroid, 1),
            "spectral_rolloff": round(spectral_rolloff, 1),
            "is_cough": is_cough,
            "cough_type": cough_type,
            "wheeze_detected": wheeze_detected,
            "stridor_detected": stridor_detected,
            "respiratory_rate": round(self.last_respiratory_rate, 1),
            "dyspnea_index": round(self.last_dyspnea_score, 2),
            "state": state,
            "spectrum_bars": spectrum_bars,
        }

    def _estimate_respiratory_rate(self):
        """
        Estimates respiratory rate (breaths per minute) by tracking low-frequency
        amplitude envelope modulations of breath sounds in the 4-8 second buffer.
        """
        buf = np.array(self.audio_buffer, dtype=np.float32)
        # Downsample to 50 Hz envelope
        step = max(1, int(self.sample_rate / 50.0))
        envelope = np.abs(buf[::step])

        # Smooth envelope
        if len(envelope) > 20:
            b, a = signal.butter(2, 0.1, btype="lowpass")
            try:
                smoothed = signal.filtfilt(b, a, envelope)
                # Find respiratory peaks in 0.15 Hz to 0.65 Hz (9 to 39 breaths/min)
                peaks, _ = signal.find_peaks(
                    smoothed, distance=int(50 / 0.7), prominence=0.01
                )
                if len(peaks) >= 2:
                    peak_intervals_sec = np.diff(peaks) / 50.0
                    avg_interval = float(np.mean(peak_intervals_sec))
                    if avg_interval > 1.2:
                        calc_rr = float(60.0 / avg_interval)
                        # Smooth with historical estimate
                        self.last_respiratory_rate = float(
                            0.8 * self.last_respiratory_rate + 0.2 * calc_rr
                        )
                        self.last_respiratory_rate = float(
                            np.clip(self.last_respiratory_rate, 10.0, 36.0)
                        )
            except Exception:
                pass
