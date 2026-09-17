"""
Baymax Healthcare Companion - Central FastAPI Server.
Exposes WebSocket telemetry stream, REST diagnostic triage APIs,
and serves the cyber-medical frontend interface.
"""

import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional
import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

# Internal module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.intelligence.medical_brain import MedicalDiagnosticBrain
from backend.intelligence.multimodal_fusion import MultimodalFusionEngine
from backend.intelligence.triage_engine import ESITriageEngine
from backend.perception.acoustic_engine import AcousticDiagnosticEngine
from backend.perception.emotion_engine import EmotionToneEngine
from backend.perception.rppg_engine import RPPGEngine
from backend.voice.speech_engine import BaymaxSpeechEngine

app = FastAPI(
    title="Baymax Healthcare Companion API",
    description="Software-only non-contact healthcare companion perception, triage, and affective interface.",
    version="1.0.0",
)

# CORS middleware for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core engines
rppg_engine = RPPGEngine(fps=30.0, buffer_seconds=6.0)
acoustic_engine = AcousticDiagnosticEngine(sample_rate=16000)
emotion_engine = EmotionToneEngine(sample_rate=16000)
triage_engine = ESITriageEngine()
fusion_engine = MultimodalFusionEngine()
medical_brain = MedicalDiagnosticBrain()
speech_engine = BaymaxSpeechEngine()

# Current shared clinical state
shared_state: Dict[str, Any] = {
    "current_symptoms": "",
    "current_pain_score": 0,
    "last_rppg": {
        "bpm": 72.0,
        "hrv": 45.0,
        "confidence": 0.0,
        "pulse_value": 0.0,
        "pulse_wave": [],
        "face_detected": False,
        "face_rect": None,
        "rois": [],
    },
    "last_acoustic": {
        "rms_energy": 0.0,
        "zcr": 0.0,
        "is_cough": False,
        "cough_type": "None",
        "wheeze_detected": False,
        "stridor_detected": False,
        "respiratory_rate": 16.0,
        "dyspnea_index": 0.0,
        "state": "Normal",
        "spectrum_bars": [0.05] * 16,
    },
    "last_emotion": {
        "stress_level": 20,
        "pain_indicator": 1,
        "panic_index": 0.1,
        "emotional_tone": "Calm & Stable",
    },
    "last_fusion": {
        "composite_health_index": 95,
        "hemodynamic_status": "Normal Resting Heart Rate",
        "pulmonary_status": "Normal Respiration",
        "clinical_correlations": [],
    },
    "last_triage": {
        "esi_level": 5,
        "severity_label": "ESI Level 5: NON-URGENT (Routine Health Advice / Self-Care)",
        "urgency_color": "#00f0ff",
        "escalation_required": False,
        "is_emergency": False,
        "clinical_rationales": ["No acute distress recorded"],
        "immediate_actions": ["Routine monitoring"],
    },
}


class TriageRequest(BaseModel):
    symptoms: str
    pain_score: int
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    pain_score: Optional[int] = None
    symptoms: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    stress_level: Optional[int] = 20
    pain_score: Optional[int] = 0
    esi_level: Optional[int] = 4


@app.get("/api/status")
def get_system_status():
    """System health check and loaded modules."""
    return {
        "status": "operational",
        "companion": "Baymax Healthcare Companion",
        "version": "1.0.0",
        "modules": {
            "rppg_vision": True,
            "acoustic_diagnostic": True,
            "affective_emotion": True,
            "esi_triage_safeguards": True,
            "multimodal_fusion": True,
            "medical_brain": True,
            "speech_synthesis": True,
        },
        "models": {
            "ollama_target": medical_brain.default_model,
            "clinical_expert_fallback": "Available",
        },
    }


@app.get("/api/vitals")
def get_current_vitals():
    """Retrieves the latest fused clinical vitals telemetry."""
    return {
        "rppg": shared_state["last_rppg"],
        "acoustic": shared_state["last_acoustic"],
        "emotion": shared_state["last_emotion"],
        "fusion": shared_state["last_fusion"],
        "triage": shared_state["last_triage"],
    }


@app.post("/api/triage")
def evaluate_triage(req: TriageRequest):
    """Executes deterministic ESI v4 clinical evaluation."""
    shared_state["current_symptoms"] = req.symptoms
    shared_state["current_pain_score"] = req.pain_score

    hr = req.heart_rate or shared_state["last_rppg"].get("bpm")
    rr = req.respiratory_rate or shared_state["last_acoustic"].get("respiratory_rate")
    acoustic_alert = shared_state["last_acoustic"].get("state")
    stress = shared_state["last_emotion"].get("stress_level")

    triage_result = triage_engine.evaluate_triage(
        symptoms_text=req.symptoms,
        pain_score=req.pain_score,
        heart_rate=hr,
        respiratory_rate=rr,
        acoustic_alert=acoustic_alert,
        stress_level=stress,
    )
    shared_state["last_triage"] = triage_result

    # Re-compute fusion with updated triage
    shared_state["last_fusion"] = fusion_engine.fuse_assessment(
        rppg_data=shared_state["last_rppg"],
        acoustic_data=shared_state["last_acoustic"],
        emotion_data=shared_state["last_emotion"],
        symptom_text=req.symptoms,
        pain_score=req.pain_score,
    )

    return {
        "triage": triage_result,
        "fusion": shared_state["last_fusion"],
    }


@app.post("/api/chat")
async def chat_with_baymax(req: ChatRequest):
    """Conversational clinical consultation with Baymax."""
    if req.pain_score is not None:
        shared_state["current_pain_score"] = req.pain_score
    if req.symptoms:
        shared_state["current_symptoms"] = req.symptoms

    # Run triage update
    triage_eval = triage_engine.evaluate_triage(
        symptoms_text=req.message or shared_state["current_symptoms"],
        pain_score=shared_state["current_pain_score"],
        heart_rate=shared_state["last_rppg"].get("bpm"),
        respiratory_rate=shared_state["last_acoustic"].get("respiratory_rate"),
        acoustic_alert=shared_state["last_acoustic"].get("state"),
        stress_level=shared_state["last_emotion"].get("stress_level"),
    )
    shared_state["last_triage"] = triage_eval

    # Multimodal fusion update
    fusion_eval = fusion_engine.fuse_assessment(
        rppg_data=shared_state["last_rppg"],
        acoustic_data=shared_state["last_acoustic"],
        emotion_data=shared_state["last_emotion"],
        symptom_text=req.message or shared_state["current_symptoms"],
        pain_score=shared_state["current_pain_score"],
    )
    shared_state["last_fusion"] = fusion_eval

    # Generate response from Medical Brain
    brain_result = await medical_brain.generate_response(
        user_message=req.message,
        triage_data=triage_eval,
        multimodal_data=fusion_eval,
    )

    # Calculate speech modulation parameters
    modulation = speech_engine.calculate_voice_modulation(
        stress_level=shared_state["last_emotion"].get("stress_level", 20),
        pain_score=shared_state["current_pain_score"],
        esi_level=triage_eval.get("esi_level", 4),
    )

    return {
        "reply": brain_result["response"],
        "source": brain_result["source"],
        "triage": triage_eval,
        "fusion": fusion_eval,
        "speech_modulation": modulation,
    }


@app.post("/api/synthesize")
def synthesize_speech(req: TTSRequest):
    """Generates audio WAV file for Baymax's speech if requested."""
    audio_path = speech_engine.synthesize_to_file(
        text=req.text,
        stress_level=req.stress_level or 20,
        pain_score=req.pain_score or 0,
        esi_level=req.esi_level or 4,
    )
    if audio_path and os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")
    # If pyttsx3 is unavailable, the frontend Web Speech API will handle speech
    raise HTTPException(
        status_code=503,
        detail="Backend pyttsx3 offline; use client-side Web Speech Synthesis.",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Bi-directional high-frequency telemetry stream.
    Receives webcam frames (base64) & audio chunks.
    Broadcasts real-time vitals, PPG waveforms, and triage telemetry.
    """
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = json.loads(raw_text)
            except Exception:
                continue

            msg_type = data.get("type")

            # 1. Incoming Video Frame for rPPG
            if msg_type == "video_frame":
                frame_b64 = data.get("image", "")
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",")[1]

                if frame_b64:
                    try:
                        img_bytes = base64.b64decode(frame_b64)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            rppg_res = rppg_engine.process_frame(frame)
                            shared_state["last_rppg"] = rppg_res
                    except Exception:
                        pass

            # 2. Incoming Audio Chunk for Acoustic Diagnostics
            elif msg_type == "audio_chunk":
                audio_samples = data.get("samples", [])
                sr = data.get("sample_rate", 16000)

                if audio_samples:
                    try:
                        audio_arr = np.array(audio_samples, dtype=np.float32)
                        # Process acoustic classification
                        acoustic_res = acoustic_engine.process_audio_chunk(
                            audio_arr, sample_rate=sr
                        )
                        shared_state["last_acoustic"] = acoustic_res

                        # Process vocal emotion & stress
                        emotion_res = emotion_engine.analyze_vocal_tone(audio_arr)
                        shared_state["last_emotion"] = emotion_res
                    except Exception:
                        pass

            # 3. Incoming Pain Score or Symptoms update
            elif msg_type == "patient_update":
                if "pain_score" in data:
                    shared_state["current_pain_score"] = int(data["pain_score"])
                if "symptoms" in data:
                    shared_state["current_symptoms"] = str(data["symptoms"])

            # Compute fusion update
            shared_state["last_fusion"] = fusion_engine.fuse_assessment(
                rppg_data=shared_state["last_rppg"],
                acoustic_data=shared_state["last_acoustic"],
                emotion_data=shared_state["last_emotion"],
                symptom_text=shared_state["current_symptoms"],
                pain_score=shared_state["current_pain_score"],
            )

            # Evaluate triage
            shared_state["last_triage"] = triage_engine.evaluate_triage(
                symptoms_text=shared_state["current_symptoms"],
                pain_score=shared_state["current_pain_score"],
                heart_rate=shared_state["last_rppg"].get("bpm"),
                respiratory_rate=shared_state["last_acoustic"].get("respiratory_rate"),
                acoustic_alert=shared_state["last_acoustic"].get("state"),
                stress_level=shared_state["last_emotion"].get("stress_level"),
            )

            # Broadcast comprehensive telemetry packet
            await websocket.send_json(
                {
                    "type": "telemetry",
                    "rppg": shared_state["last_rppg"],
                    "acoustic": shared_state["last_acoustic"],
                    "emotion": shared_state["last_emotion"],
                    "fusion": shared_state["last_fusion"],
                    "triage": shared_state["last_triage"],
                }
            )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Serve frontend static files
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Baymax UI loading..."}
