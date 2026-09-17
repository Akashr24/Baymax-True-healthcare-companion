"""
Unit tests for Acoustic Diagnostic Engine.
Tests spectral feature extraction, cough burst classification, and wheeze detection.
"""

import unittest
import numpy as np
from backend.perception.acoustic_engine import AcousticDiagnosticEngine


class TestAcousticEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AcousticDiagnosticEngine(sample_rate=16000)

    def test_silence_detection(self):
        """Tests that silence yields resting breath baseline."""
        silence = np.zeros(16000, dtype=np.float32)
        res = self.engine.process_audio_chunk(silence)
        self.assertFalse(res["is_cough"])
        self.assertFalse(res["wheeze_detected"])
        self.assertLess(res["rms_energy"], 0.02)

    def test_wheeze_detection(self):
        """Simulates continuous high-energy 650 Hz musical wheezing tone."""
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        # Strong 650 Hz tone + low background noise
        wheeze_signal = 0.3 * np.sin(2 * np.pi * 650.0 * t) + 0.02 * np.random.randn(sr)
        res = self.engine.process_audio_chunk(wheeze_signal.astype(np.float32))

        self.assertTrue(res["wheeze_detected"])
        self.assertIn("Wheez", res["state"])

    def test_cough_burst_detection(self):
        """Simulates a sudden high-energy wideband cough burst."""
        sr = 16000
        # 200ms sudden burst of noisy high frequency sound
        cough_burst = np.zeros(sr, dtype=np.float32)
        burst_samples = int(sr * 0.25)
        # High frequency noise centered around 2500 Hz
        t = np.linspace(0, 0.25, burst_samples)
        noise = np.random.randn(burst_samples) * np.hanning(burst_samples) * 0.4
        cough_burst[:burst_samples] = noise

        res = self.engine.process_audio_chunk(cough_burst)
        self.assertTrue(res["is_cough"])
        self.assertNotEqual(res["cough_type"], "None")


if __name__ == "__main__":
    unittest.main()
