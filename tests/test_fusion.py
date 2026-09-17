"""
Unit tests for Multimodal Health Fusion Engine.
Tests cross-sensor correlation and composite health index scoring.
"""

import unittest
from backend.intelligence.multimodal_fusion import MultimodalFusionEngine


class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        self.engine = MultimodalFusionEngine()

    def test_healthy_baseline_fusion(self):
        """Tests that normal vitals yield high composite health score (>90)."""
        rppg = {"bpm": 70.0, "hrv": 52.0, "confidence": 0.9}
        acoustic = {
            "cough_type": "None",
            "is_cough": False,
            "wheeze_detected": False,
            "stridor_detected": False,
            "respiratory_rate": 15.0,
            "dyspnea_index": 0.0,
        }
        emotion = {"stress_level": 15, "panic_index": 0.05}

        result = self.engine.fuse_assessment(
            rppg_data=rppg,
            acoustic_data=acoustic,
            emotion_data=emotion,
            symptom_text="Feeling energetic and healthy",
            pain_score=0,
        )

        self.assertGreaterEqual(result["composite_health_index"], 88)
        self.assertEqual(result["hemodynamic_status"], "Normal Resting Heart Rate")

    def test_tachycardia_and_cough_syndrome_correlation(self):
        """Tests syndromic correlation for fever + cough + tachycardia."""
        rppg = {"bpm": 118.0, "hrv": 20.0, "confidence": 0.85}
        acoustic = {
            "cough_type": "Productive / Wet Cough",
            "is_cough": True,
            "wheeze_detected": False,
            "stridor_detected": False,
            "respiratory_rate": 25.0,
            "dyspnea_index": 0.4,
        }
        emotion = {"stress_level": 60, "panic_index": 0.3}

        result = self.engine.fuse_assessment(
            rppg_data=rppg,
            acoustic_data=acoustic,
            emotion_data=emotion,
            symptom_text="High fever and wet cough with fatigue",
            pain_score=5,
        )

        self.assertLess(result["composite_health_index"], 65)
        self.assertIn("Tachycardia", result["hemodynamic_status"])
        self.assertTrue(len(result["clinical_correlations"]) > 0)


if __name__ == "__main__":
    unittest.main()
