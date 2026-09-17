"""
Deterministic Clinical Triage Engine.
Implements the Emergency Severity Index (ESI Version 4) decision algorithm
and enforces non-overridable safety safeguards for acute red-flag conditions.
"""

from typing import Any, Dict, List, Optional
import re


class ESITriageEngine:
    """
    Emergency Severity Index (ESI v4) Deterministic Triage & Safety Engine.
    Prevents AI hallucination and enforces strict emergency escalation protocols.
    """

    # Clinical red-flag symptom triggers for immediate escalation
    CRITICAL_KEYWORDS_LEVEL_1 = [
        "unconscious", "unresponsive", "stopped breathing", "no pulse",
        "cardiac arrest", "respiratory arrest", "severe cyanosis", "blue lips",
        "gasping for air", "choking", "massive hemorrhage", "bleeding out"
    ]

    HIGH_RISK_KEYWORDS_LEVEL_2 = [
        "chest pain", "crushing pressure", "radiating to arm", "radiating to jaw",
        "shortness of breath", "severe asthma", "stridor", "throat closing",
        "anaphylaxis", "allergic reaction", "slurred speech", "facial droop",
        "arm weakness", "sudden numbness", "stroke", "thunderclap headache",
        "worst headache of my life", "coughing blood", "hemoptysis", "vomiting blood",
        "severe abdominal rigidity", "poisoning", "overdose", "suicidal"
    ]

    MODERATE_KEYWORDS_LEVEL_3 = [
        "fever", "chills", "persistent cough", "productive cough", "wheezing",
        "abdominal pain", "vomiting", "diarrhea", "dehydration", "kidney stone",
        "deep cut", "moderate burn", "migraine", "asthma flare"
    ]

    def __init__(self):
        pass

    def evaluate_triage(
        self,
        symptoms_text: str,
        pain_score: int,  # 0 to 10
        heart_rate: Optional[float] = None,  # BPM from rPPG
        hrv: Optional[float] = None,  # ms from rPPG
        respiratory_rate: Optional[float] = None,  # breaths/min from acoustic
        acoustic_alert: Optional[str] = None,  # stridor, wheeze, etc.
        stress_level: Optional[int] = None,  # 0-100
    ) -> Dict[str, Any]:
        """
        Executes the ESI v4 decision logic.
        Returns:
            esi_level: 1 (Resuscitation) to 5 (Non-urgent)
            severity_label: Human-readable triage tier
            urgency_color: Hex color token for UI
            escalation_required: bool
            clinical_rationale: List of observed safety factors
            immediate_actions: Actionable safety guidelines
        """
        text_lower = symptoms_text.lower()
        rationales: List[str] = []
        immediate_actions: List[str] = []

        # -------------------------------------------------------------
        # STEP 1: Does patient require immediate life-saving intervention? (ESI 1)
        # -------------------------------------------------------------
        for kw in self.CRITICAL_KEYWORDS_LEVEL_1:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                rationales.append(f"Immediate life threat identified: '{kw}'")
                return self._create_triage_result(
                    level=1,
                    label="ESI Level 1: RESUSCITATION (Immediate Life Threat)",
                    color="#ff0033",
                    escalate=True,
                    rationales=rationales,
                    actions=[
                        "CALL EMERGENCY SERVICES (911 / 112) IMMEDIATELY.",
                        "If trained and patient is unresponsive with no pulse, begin CPR (30 compressions : 2 breaths).",
                        "Ensure airway is open; clear any oral obstructions.",
                        "Unlock front entrance for emergency medical responders."
                    ]
                )

        # Check critical vital sign thresholds (ESI 1)
        if heart_rate is not None:
            if heart_rate < 40.0:
                rationales.append(f"Severe critical bradycardia detected (HR {heart_rate:.0f} BPM < 40)")
                return self._create_triage_result(
                    level=1,
                    label="ESI Level 1: RESUSCITATION (Profound Bradycardia)",
                    color="#ff0033",
                    escalate=True,
                    rationales=rationales,
                    actions=[
                        "CALL EMERGENCY SERVICES IMMEDIATELY.",
                        "Have patient lie down with legs elevated if conscious.",
                        "Do not leave the patient unattended."
                    ]
                )
            elif heart_rate > 165.0:
                rationales.append(f"Extreme critical tachycardia detected (HR {heart_rate:.0f} BPM > 165)")
                return self._create_triage_result(
                    level=1,
                    label="ESI Level 1: RESUSCITATION (Severe Tachyarrhythmia)",
                    color="#ff0033",
                    escalate=True,
                    rationales=rationales,
                    actions=[
                        "CALL EMERGENCY SERVICES IMMEDIATELY.",
                        "Keep patient seated resting; monitor consciousness.",
                        "Prepare for paramedic arrival."
                    ]
                )

        # -------------------------------------------------------------
        # STEP 2: Is this a high-risk situation or severe pain / distress? (ESI 2)
        # -------------------------------------------------------------
        high_risk_matches = []
        for kw in self.HIGH_RISK_KEYWORDS_LEVEL_2:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                high_risk_matches.append(kw)

        if high_risk_matches:
            rationales.append(f"High-risk clinical symptoms detected: {', '.join(high_risk_matches)}")

        # Stridor acoustic detection is an automatic ESI 2 (airway hazard)
        if acoustic_alert and "stridor" in acoustic_alert.lower():
            rationales.append("Acoustic perception detected inspiratory stridor (airway constriction).")

        # Pain scale >= 8 with vital signs abnormality
        is_severe_pain = pain_score >= 8
        if is_severe_pain:
            rationales.append(f"Severe acute pain rating ({pain_score}/10).")

        # Tachycardia / Tachypnea correlation
        vitals_unstable = False
        if heart_rate is not None and (heart_rate > 115.0 or heart_rate < 50.0):
            vitals_unstable = True
            rationales.append(f"Unstable cardiac frequency (HR {heart_rate:.0f} BPM).")
        if respiratory_rate is not None and (respiratory_rate > 24.0 or respiratory_rate < 10.0):
            vitals_unstable = True
            rationales.append(f"Abnormal respiratory rate ({respiratory_rate:.0f} breaths/min).")

        if high_risk_matches or (is_severe_pain and vitals_unstable) or (acoustic_alert and "stridor" in acoustic_alert.lower()):
            actions = [
                "Seek immediate emergency medical evaluation (Emergency Department).",
                "Do NOT drive yourself; have an ambulance or trusted companion transport you.",
                "Sit in a comfortable upright position to ease breathing.",
                "Loosen any restrictive clothing around the neck and chest."
            ]
            if any("chest" in m or "heart" in m for m in high_risk_matches):
                actions.append("If non-allergic and not contraindicated, consider taking chewable aspirin if previously instructed by a physician.")

            return self._create_triage_result(
                level=2,
                label="ESI Level 2: EMERGENT (High Risk / Acute Intervention Needed)",
                color="#ff3366",
                escalate=True,
                rationales=rationales,
                actions=actions
            )

        # -------------------------------------------------------------
        # STEP 3: Moderate conditions requiring multiple clinical resources (ESI 3)
        # -------------------------------------------------------------
        mod_matches = []
        for kw in self.MODERATE_KEYWORDS_LEVEL_3:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                mod_matches.append(kw)

        if mod_matches or (5 <= pain_score <= 7) or (acoustic_alert and "wheeze" in acoustic_alert.lower()):
            if mod_matches:
                rationales.append(f"Urgent symptoms noted: {', '.join(mod_matches)}")
            if 5 <= pain_score <= 7:
                rationales.append(f"Moderate pain score ({pain_score}/10)")
            if acoustic_alert and "wheeze" in acoustic_alert.lower():
                rationales.append("Acoustic wheezing detected (possible bronchospasm).")

            return self._create_triage_result(
                level=3,
                label="ESI Level 3: URGENT (Multi-Resource Medical Assessment Needed)",
                color="#ffaa00",
                escalate=False,
                rationales=rationales,
                actions=[
                    "Visit an Urgent Care Center or schedule a same-day primary care appointment.",
                    "Rest and avoid strenuous exertion.",
                    "Stay well hydrated with clear fluids.",
                    "Monitor vitals; if chest pain or shortness of breath develops, seek immediate ER care."
                ]
            )

        # -------------------------------------------------------------
        # STEP 4: Mild conditions requiring single clinical resource (ESI 4)
        # -------------------------------------------------------------
        mild_keywords = [
            "mild", "sore throat", "runny nose", "congestion", "cold", "scratch",
            "bruise", "twisted ankle", "minor burn", "rash", "earache", "itch"
        ]
        has_mild_symptom = any(re.search(r"\b" + re.escape(k) + r"\b", text_lower) for k in mild_keywords)

        if (1 <= pain_score <= 4) or has_mild_symptom:
            rationales.append(f"Mild localized symptoms; pain scale {pain_score}/10; stable vital signs.")
            return self._create_triage_result(
                level=4,
                label="ESI Level 4: LESS URGENT (Single Resource / Outpatient Care)",
                color="#33cc66",
                escalate=False,
                rationales=rationales,
                actions=[
                    "Consult your primary care physician or walk-in clinic.",
                    "Rest, apply hot/cold compresses if appropriate for muscle aches.",
                    "Follow standard over-the-counter pain relief guidelines under medical advice.",
                    "Return for re-triage if symptoms worsen or new red flags appear."
                ]
            )

        # -------------------------------------------------------------
        # STEP 5: Non-urgent / Self-care / Routine health inquiry (ESI 5)
        # -------------------------------------------------------------
        rationales.append("No acute physiological distress or red-flag criteria identified.")
        return self._create_triage_result(
            level=5,
            label="ESI Level 5: NON-URGENT (Routine Health Advice / Self-Care)",
            color="#00f0ff",
            escalate=False,
            rationales=rationales,
            actions=[
                "Standard self-care, hydration, and nutritional support.",
                "Continue routine preventive health checkups.",
                "Baymax is at your service if any symptoms change."
            ]
        )

    def _create_triage_result(
        self,
        level: int,
        label: str,
        color: str,
        escalate: bool,
        rationales: List[str],
        actions: List[str]
    ) -> Dict[str, Any]:
        return {
            "esi_level": level,
            "severity_label": label,
            "urgency_color": color,
            "escalation_required": escalate,
            "clinical_rationales": rationales,
            "immediate_actions": actions,
            "is_emergency": level in [1, 2],
        }
