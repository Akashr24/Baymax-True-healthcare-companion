"""
Multimodal Health Fusion Engine.
Correlates optical vitals (rPPG Heart Rate & HRV), acoustic biomarkers (cough, wheeze, breath rate),
affective vocal cues (stress, panic), and patient-reported symptoms & pain scale.
"""

from typing import Any, Dict, List, Optional


class MultimodalFusionEngine:
    """
    Fuses cross-sensory physiological signals and patient reports into a
    coherent clinical profile with composite health indices.
    """

    def __init__(self):
        pass

    def fuse_assessment(
        self,
        rppg_data: Dict[str, Any],
        acoustic_data: Dict[str, Any],
        emotion_data: Dict[str, Any],
        symptom_text: str,
        pain_score: int,
    ) -> Dict[str, Any]:
        """
        Fuses multimodal sensor data into a structured clinical picture.
        """
        bpm = rppg_data.get("bpm", 72.0)
        hrv = rppg_data.get("hrv", 45.0)
        bpm_confidence = rppg_data.get("confidence", 0.5)

        cough_type = acoustic_data.get("cough_type", "None")
        is_cough = acoustic_data.get("is_cough", False)
        wheeze = acoustic_data.get("wheeze_detected", False)
        stridor = acoustic_data.get("stridor_detected", False)
        resp_rate = acoustic_data.get("respiratory_rate", 16.0)
        dyspnea = acoustic_data.get("dyspnea_index", 0.0)

        stress_level = emotion_data.get("stress_level", 20)
        panic_index = emotion_data.get("panic_index", 0.1)

        # 1. Hemodynamic Status
        if bpm > 115.0:
            hemodynamic_status = "Tachycardia (Elevated Heart Rate)"
        elif bpm < 55.0:
            hemodynamic_status = "Bradycardia (Low Heart Rate)"
        else:
            hemodynamic_status = "Normal Resting Heart Rate"

        if hrv < 22.0:
            autonomic_state = "Sympathetic Dominance / High Physiological Stress"
        elif hrv > 70.0:
            autonomic_state = "High Parasympathetic Tone / Well Rested"
        else:
            autonomic_state = "Balanced Autonomic Function"

        # 2. Pulmonary Status
        if stridor:
            pulmonary_status = "CRITICAL: Upper Airway Obstruction (Inspiratory Stridor)"
        elif wheeze:
            pulmonary_status = "Bronchial Constriction / Wheezing"
        elif is_cough and "Productive" in cough_type:
            pulmonary_status = "Lower Airway Secretions / Productive Cough"
        elif is_cough and "Dry" in cough_type:
            pulmonary_status = "Airway Irritation / Dry Spasmodic Cough"
        elif resp_rate > 24.0:
            pulmonary_status = "Tachypnea (Rapid Respiratory Rate)"
        else:
            pulmonary_status = "Eupnea (Normal Respiration)"

        # 3. Composite Health Index (0 - 100)
        health_score = 100.0

        # Deductions based on objective deviations
        if bpm > 100:
            health_score -= min(25.0, (bpm - 100) * 0.7)
        elif bpm < 55:
            health_score -= min(20.0, (55 - bpm) * 0.8)

        if resp_rate > 20:
            health_score -= min(20.0, (resp_rate - 20) * 1.5)

        if wheeze:
            health_score -= 20.0
        if stridor:
            health_score -= 35.0
        if is_cough:
            health_score -= 10.0

        # Pain impact
        health_score -= pain_score * 3.5

        # Stress/Panic impact
        health_score -= (stress_level / 100.0) * 15.0

        composite_health_index = int(np_clip(health_score, 10, 100))

        # 4. Cross-Modality Clinical Syndromic Correlations
        clinical_flags: List[str] = []
        sym_lower = symptom_text.lower()

        # Cardiac / Ischemic correlation
        if ("chest" in sym_lower or "pressure" in sym_lower) and (bpm > 100 or pain_score >= 7):
            clinical_flags.append(
                "Potential Acute Coronary Syndrome: Patient reports chest symptoms with elevated cardiac strain."
            )

        # Respiratory Infection correlation
        if ("fever" in sym_lower or "chills" in sym_lower) and (is_cough or wheeze or resp_rate > 22):
            clinical_flags.append(
                "Suspected Lower Respiratory Tract Infection: Combination of thermal complaints and acoustic pulmonary findings."
            )

        # Asthma / Bronchospasm correlation
        if wheeze and ("asthma" in sym_lower or "tight" in sym_lower or dyspnea > 0.4):
            clinical_flags.append(
                "Acute Bronchospasm / Asthma Exacerbation: Audible expiratory wheezing with respiratory workload."
            )

        # Hyperventilation / Panic Attack correlation
        if panic_index > 0.6 and resp_rate > 24 and pain_score <= 4 and "chest" not in sym_lower:
            clinical_flags.append(
                "Psychophysiological Hyperventilation / Acute Anxiety: Elevated vocal stress and respiratory pacing without focal organic red-flags."
            )

        return {
            "composite_health_index": composite_health_index,
            "hemodynamic_status": hemodynamic_status,
            "autonomic_state": autonomic_state,
            "pulmonary_status": pulmonary_status,
            "clinical_correlations": clinical_flags,
            "vitals_summary": {
                "heart_rate_bpm": round(bpm, 1),
                "hrv_ms": round(hrv, 1),
                "respiratory_rate": round(resp_rate, 1),
                "dyspnea_score": round(dyspnea, 2),
                "pain_score": pain_score,
                "stress_level": stress_level,
                "confidence": round(bpm_confidence, 2),
            },
        }


def np_clip(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))
