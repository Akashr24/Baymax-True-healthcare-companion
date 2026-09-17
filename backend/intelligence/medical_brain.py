"""
Local Medical Diagnostic Brain.
Provides clinical reasoning, differential diagnosis formulation, and empathic guidance.
Architecture supports:
1. Local Ollama (BioMistral-7B, MedLlama-3, etc.)
2. Deterministic Clinical Expert Knowledge Base (zero-latency offline clinical brain)
3. Cloud Fallback (Google Gemini / OpenAI) if configured.
"""

import os
from typing import Any, Dict, List, Optional
import httpx


class MedicalDiagnosticBrain:
    """
    Diagnostic reasoning engine for Baymax.
    Synthesizes symptoms, physiological telemetry, and clinical triage criteria
    into compassionate, medically grounded responses.
    """

    BAYMAX_SYSTEM_PROMPT = """You are Baymax, the personal healthcare companion from Big Hero 6.
You speak with a calm, gentle, reassuring, and mildly formal tone. You are deeply attentive to the patient's well-being.
Your primary directive is patient safety, comfort, and accurate clinical guidance.
Always take into account the patient's measured vitals (Heart Rate, Respiratory Rate, HRV, Acoustic cough findings) and pain rating.
Never give definitive definitive diagnoses without advising consultation with a licensed human physician.
If the case is ESI Level 1 or 2, immediately instruct the patient to call emergency services or go to the nearest emergency department.
Always end your interaction with an empathetic offer of continued care."""

    def __init__(self, ollama_url: str = "http://localhost:11434", default_model: str = "biomistral"):
        self.ollama_url = os.environ.get("OLLAMA_URL", ollama_url)
        self.default_model = os.environ.get("OLLAMA_MODEL", default_model)
        self.gemini_key = os.environ.get("GEMINI_API_KEY")

        # Built-in Clinical Diagnostic Knowledge Tree
        self.clinical_knowledge_base = self._init_clinical_knowledge()

    def _init_clinical_knowledge(self) -> Dict[str, Any]:
        """Initializes deterministic clinical syndromes and guidance."""
        return {
            "cardiac": {
                "keywords": ["chest pain", "chest pressure", "heart", "left arm", "jaw pain", "angina"],
                "differentials": ["Acute Coronary Syndrome", "Angina Pectoris", "Costochondritis", "Gastroesophageal Reflux"],
                "baymax_advice": (
                    "I notice you are experiencing discomfort in your chest. Because your heart is a vital organ, "
                    "chest discomfort must always be evaluated immediately by medical professionals. Please sit resting, "
                    "loosen tight clothing, and do not exert yourself while emergency care is contacted."
                )
            },
            "respiratory": {
                "keywords": ["cough", "wheeze", "breathing", "phlegm", "short of breath", "asthma", "bronchitis"],
                "differentials": ["Acute Bronchitis", "Bronchial Asthma Exacerbation", "Pneumonia", "Upper Respiratory Infection"],
                "baymax_advice": (
                    "My acoustic sensors detect respiratory congestion. Maintaining upright posture helps expand your lung capacity. "
                    "Warm hydration and steam inhalation can soothe irritated bronchial passages. If your breathing becomes labored or your lips appear dusky, "
                    "seek urgent medical intervention immediately."
                )
            },
            "fever_infection": {
                "keywords": ["fever", "chills", "sweating", "infection", "body aches", "flu"],
                "differentials": ["Viral Syndrome / Influenza", "Bacterial Infection", "Systemic Inflammatory Response"],
                "baymax_advice": (
                    "Your symptoms are consistent with an active immune response. I recommend adequate hydration, restful sleep, "
                    "and light clothing to assist thermoregulation. If fever exceeds 39.4°C (103°F) or does not respond to antipyretics, "
                    "a clinical examination is warranted."
                )
            },
            "gastrointestinal": {
                "keywords": ["stomach", "nausea", "vomiting", "diarrhea", "cramp", "belly"],
                "differentials": ["Gastroenteritis", "Acid Peptic Disease", "Foodborne Illness", "Irritable Bowel Syndrome"],
                "baymax_advice": (
                    "Gastrointestinal distress can deplete essential fluids and electrolytes. Sip oral rehydration solutions slowly in small amounts. "
                    "Avoid spicy, greasy, or dairy products for 24 hours. Seek immediate care if you develop severe localized pain in the lower right abdomen or signs of blood."
                )
            },
            "headache_neurological": {
                "keywords": ["headache", "migraine", "head throbbing", "dizzy", "lightheaded"],
                "differentials": ["Tension-Type Headache", "Migraine with/without Aura", "Dehydration Headache", "Orthostatic Hypotension"],
                "baymax_advice": (
                    "I suggest resting in a dimly lit, quiet room. Applying a cold or warm compress to the forehead or neck often relieves muscular tension. "
                    "Ensure you have consumed adequate water today. A sudden, thunderclap headache requires immediate emergency evaluation."
                )
            },
            "musculoskeletal": {
                "keywords": ["back pain", "sprain", "muscle", "joint", "shoulder", "knee", "ache"],
                "differentials": ["Acute Muscle Strain", "Ligamentous Sprain", "Tendinopathy", "Myofascial Pain"],
                "baymax_advice": (
                    "I recommend the R.I.C.E. protocol: Rest the affected area, apply Ice packs wrapped in cloth for 15 minutes, "
                    "apply gentle Compression if swollen, and Elevate the limb. Avoid motions that aggravate your pain."
                )
            }
        }

    async def generate_response(
        self,
        user_message: str,
        triage_data: Dict[str, Any],
        multimodal_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates clinical reasoning and dialogue. Tries Ollama first, then Cloud LLM,
        and seamlessly falls back to the deterministic Clinical Knowledge Engine.
        """
        esi_level = triage_data.get("esi_level", 4)
        is_emergency = triage_data.get("is_emergency", False)

        # 1. First priority: Try Local Ollama if available
        ollama_response = await self._query_ollama(
            user_message, triage_data, multimodal_data, conversation_history
        )
        if ollama_response:
            return {
                "source": "Local Medical LLM (Ollama)",
                "response": ollama_response,
                "triage_level": esi_level,
                "is_emergency": is_emergency,
            }

        # 2. Second: Try Google Gemini if API Key is configured
        if self.gemini_key:
            gemini_response = await self._query_gemini(
                user_message, triage_data, multimodal_data
            )
            if gemini_response:
                return {
                    "source": "Gemini Clinical Reasoning Engine",
                    "response": gemini_response,
                    "triage_level": esi_level,
                    "is_emergency": is_emergency,
                }

        # 3. Deterministic Clinical Knowledge Engine (Always available, 0ms latency, safe)
        expert_response = self._generate_expert_clinical_guidance(
            user_message, triage_data, multimodal_data
        )
        return {
            "source": "Baymax Deterministic Clinical Engine",
            "response": expert_response,
            "triage_level": esi_level,
            "is_emergency": is_emergency,
        }

    def _generate_expert_clinical_guidance(
        self,
        user_message: str,
        triage_data: Dict[str, Any],
        multimodal_data: Dict[str, Any]
    ) -> str:
        """Constructs an empathic, medically grounded Baymax response."""
        esi_level = triage_data.get("esi_level", 5)
        severity_label = triage_data.get("severity_label", "Non-urgent")
        vitals = multimodal_data.get("vitals_summary", {})
        bpm = vitals.get("heart_rate_bpm", 72.0)
        rr = vitals.get("respiratory_rate", 16.0)
        pain = vitals.get("pain_score", 0)

        # Handle Emergency Severity Levels 1 & 2
        if esi_level in [1, 2]:
            actions_text = "\n• " + "\n• ".join(triage_data.get("immediate_actions", []))
            return (
                f"Hello, I am Baymax, your personal healthcare companion. "
                f"My sensors indicate that your condition requires immediate urgent medical care. "
                f"Your vital signs reflect an elevated heart rate of {bpm:.0f} BPM and a high pain rating of {pain}/10. "
                f"\n\n**CRITICAL MEDICAL PROTOCOL ACTIVATED ({severity_label})**:\n{actions_text}\n\n"
                f"Please do not exert yourself. Help is being prioritized for your safety. "
                f"I will remain active and monitor your vitals until medical assistance arrives."
            )

        # Identify clinical syndrome from knowledge base
        text_lower = user_message.lower()
        matched_advice = None
        matched_differentials = []

        for category, data in self.clinical_knowledge_base.items():
            for kw in data["keywords"]:
                if kw in text_lower:
                    matched_advice = data["baymax_advice"]
                    matched_differentials = data["differentials"]
                    break
            if matched_advice:
                break

        # If no specific syndrome keyword matched
        if not matched_advice:
            matched_advice = (
                "I have recorded your symptom history and vital signs. Your heart rate is currently "
                f"{bpm:.0f} BPM, your respiratory rate is approximately {rr:.0f} breaths per minute, "
                f"and you indicated a pain rating of {pain}/10. I recommend rest, staying properly hydrated, "
                "and discussing these symptoms with your physician if they persist."
            )
            matched_differentials = ["General Malaise", "Functional Fatigue / Mild Somatic Discomfort"]

        diff_str = ", ".join(matched_differentials[:3])

        response_parts = [
            f"Hello, I am Baymax, your personal healthcare companion.",
            f"I have scanned your physiological indicators. Based on your reported symptoms and current vital readings "
            f"(Heart Rate: {bpm:.0f} BPM, Respiration: {rr:.0f} BPM, Pain Scale: {pain}/10), "
            f"your triage assessment is categorized as **{severity_label}**.",
            f"\n**Clinical Assessment:**\nPotential differential considerations for your physician to investigate include: *{diff_str}*.",
            f"\n**Care Guidance:**\n{matched_advice}",
            "\n*I will continue monitoring your vital signs. If your pain intensifies or new symptoms appear, please alert me immediately.*"
        ]

        return "\n\n".join(response_parts)

    async def _query_ollama(
        self,
        user_message: str,
        triage_data: Dict[str, Any],
        multimodal_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Queries local Ollama endpoint if running."""
        try:
            vitals = multimodal_data.get("vitals_summary", {})
            context_prompt = (
                f"{self.BAYMAX_SYSTEM_PROMPT}\n\n"
                f"CURRENT PATIENT VITALS TELEMETRY:\n"
                f"- Heart Rate: {vitals.get('heart_rate_bpm', 72)} BPM\n"
                f"- Respiratory Rate: {vitals.get('respiratory_rate', 16)} breaths/min\n"
                f"- HRV (RMSSD): {vitals.get('hrv_ms', 45)} ms\n"
                f"- Reported Pain Score: {vitals.get('pain_score', 0)} / 10\n"
                f"- ESI Triage Level: {triage_data.get('esi_level', 4)} ({triage_data.get('severity_label', 'Less Urgent')})\n"
                f"- Acoustic Cough Findings: {multimodal_data.get('pulmonary_status', 'Normal')}\n\n"
                f"PATIENT REPORT:\n{user_message}\n\n"
                f"BAYMAX HEALTHCARE RESPONSE:"
            )

            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.default_model,
                        "prompt": context_prompt,
                        "stream": False,
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "").strip()
        except Exception:
            return None
        return None

    async def _query_gemini(
        self,
        user_message: str,
        triage_data: Dict[str, Any],
        multimodal_data: Dict[str, Any]
    ) -> Optional[str]:
        """Queries Google Gemini API if key is set."""
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            vitals = multimodal_data.get("vitals_summary", {})
            prompt = (
                f"{self.BAYMAX_SYSTEM_PROMPT}\n\n"
                f"TELEMETRY: HR={vitals.get('heart_rate_bpm')} BPM, Pain={vitals.get('pain_score')}/10, "
                f"ESI={triage_data.get('esi_level')} ({triage_data.get('severity_label')}), "
                f"Acoustic={multimodal_data.get('pulmonary_status')}.\n"
                f"User: {user_message}\nBaymax Response:"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception:
            return None
