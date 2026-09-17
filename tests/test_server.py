"""
Integration tests for FastAPI endpoints using TestClient.
"""

import unittest
from fastapi.testclient import TestClient
from backend.app import app


class TestServerEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "operational")
        self.assertTrue(data["modules"]["rppg_vision"])
        self.assertTrue(data["modules"]["esi_triage_safeguards"])

    def test_vitals_endpoint(self):
        res = self.client.get("/api/vitals")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("rppg", data)
        self.assertIn("acoustic", data)
        self.assertIn("triage", data)

    def test_triage_endpoint_emergency(self):
        payload = {
            "symptoms": "Severe crushing chest pain radiating down left arm",
            "pain_score": 9,
            "heart_rate": 130.0,
            "respiratory_rate": 28.0,
        }
        res = self.client.post("/api/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["triage"]["is_emergency"])
        self.assertEqual(data["triage"]["esi_level"], 2)

    def test_chat_endpoint(self):
        payload = {
            "message": "I have had a mild headache and tiredness since this morning",
            "pain_score": 3,
        }
        res = self.client.post("/api/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reply", data)
        self.assertIn("Baymax", data["reply"])
        self.assertIn("speech_modulation", data)
        self.assertIn("rate", data["speech_modulation"])

    def test_frontend_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Baymax", res.text)
        self.assertIn("rPPG", res.text)


if __name__ == "__main__":
    unittest.main()
