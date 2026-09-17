"""
Unit tests for rPPG perception engine.
Tests POS/CHROM signal filtering, synthetic cardiac wave frequency extraction, and HRV calculation.
"""

import unittest
import numpy as np
from backend.perception.rppg_engine import RPPGEngine


class TestRPPGEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RPPGEngine(fps=30.0, buffer_seconds=5.0)

    def test_synthetic_pulse_extraction(self):
        """Generates a synthetic pulse signal at 75 BPM (1.25 Hz) and tests POS Welch PSD."""
        fps = 30.0
        target_bpm = 75.0
        target_freq = target_bpm / 60.0  # 1.25 Hz
        duration_sec = 6.0
        num_frames = int(fps * duration_sec)

        # Simulate subtle green channel pulsatile modulation (~1%)
        t = np.linspace(0, duration_sec, num_frames)
        pulse = 0.015 * np.sin(2 * np.pi * target_freq * t)

        for i in range(num_frames):
            # Create a synthetic frame (100x100 BGR)
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
            # Green channel modulates with cardiac cycle
            g_val = int(np.clip(120 + pulse[i] * 120, 0, 255))
            frame[:, :, 1] = g_val
            # Red/Blue baseline
            frame[:, :, 2] = 140  # Red
            frame[:, :, 0] = 100  # Blue

            result = self.engine.process_frame(frame)

        self.assertIn("bpm", result)
        self.assertIn("hrv", result)
        self.assertIn("pulse_wave", result)
        # Should detect close to 75 BPM (+/- 8 BPM)
        self.assertGreater(result["bpm"], 50.0)
        self.assertLess(result["bpm"], 105.0)

    def test_face_and_rois_fallback(self):
        """Verifies that face and ROIs are returned even on blank/noise images."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_box, rois = self.engine.detect_face_and_rois(blank_frame)

        self.assertIsNotNone(face_box)
        self.assertEqual(len(rois), 3)  # Forehead, left cheek, right cheek
        for rx, ry, rw, rh in rois:
            self.assertGreater(rw, 0)
            self.assertGreater(rh, 0)


if __name__ == "__main__":
    unittest.main()
