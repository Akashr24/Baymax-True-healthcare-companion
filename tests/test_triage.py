"""
Unit tests for Deterministic ESI v4 Triage Engine.
Verifies emergency escalation safeguards and deterministic triage levels.
"""

import unittest
from backend.intelligence.triage_engine import ESITriageEngine


class TestTriageEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ESITriageEngine()

    def test_esi_level_1_resuscitation(self):
        """Cardiac arrest or unresponsiveness must immediately yield ESI Level 1."""
        res = self.engine.evaluate_triage(
            symptoms_text="Patient is unconscious and not breathing, no pulse found",
            pain_score=0
        )
        self.assertEqual(res["esi_level"], 1)
        self.assertTrue(res["escalation_required"])
        self.assertTrue(res["is_emergency"])
        self.assertIn("RESUSCITATION", res["severity_label"])

    def test_esi_level_2_acute_chest_pain(self):
        """Crushing chest pain must trigger ESI Level 2 emergency escalation."""
        res = self.engine.evaluate_triage(
            symptoms_text="Crushing chest pain radiating to left arm and jaw",
            pain_score=9,
            heart_rate=128.0,
            respiratory_rate=26.0
        )
        self.assertEqual(res["esi_level"], 2)
        self.assertTrue(res["escalation_required"])
        self.assertTrue(res["is_emergency"])
        self.assertIn("EMERGENT", res["severity_label"])

    def test_esi_level_2_stridor_airway(self):
        """Inspiratory stridor is an airway hazard requiring ESI Level 2."""
        res = self.engine.evaluate_triage(
            symptoms_text="Throat feels tight, difficulty inhaling",
            pain_score=4,
            acoustic_alert="CRITICAL: Inspiratory Stridor / Airway Obstruction"
        )
        self.assertEqual(res["esi_level"], 2)
        self.assertTrue(res["escalation_required"])

    def test_esi_level_3_urgent(self):
        """Fever and productive cough without shock is ESI Level 3."""
        res = self.engine.evaluate_triage(
            symptoms_text="High fever, chills, and persistent productive cough for 3 days",
            pain_score=6,
            heart_rate=88.0,
            respiratory_rate=18.0
        )
        self.assertEqual(res["esi_level"], 3)
        self.assertFalse(res["escalation_required"])

    def test_esi_level_5_non_urgent(self):
        """Routine inquiry with 0 pain is ESI Level 5."""
        res = self.engine.evaluate_triage(
            symptoms_text="Routine health question about sleep and nutrition",
            pain_score=0,
            heart_rate=72.0
        )
        self.assertEqual(res["esi_level"], 5)
        self.assertFalse(res["escalation_required"])


if __name__ == "__main__":
    unittest.main()
