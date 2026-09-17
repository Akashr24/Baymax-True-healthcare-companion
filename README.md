# Baymax: True Healthcare Companion

> *"Hello, I am Baymax, your personal healthcare companion."*

A software-first healthcare companion that implements non-contact physiological perception, deterministic clinical triage, and an empathic conversational interface.

---

## System Architecture

```
 ┌─────────────────────────────────────────────────────────┐
 │                   INPUT PERCEPTION LAYER                 │
 │  • Video Stream: rPPG Heart Rate & HRV (POS & CHROM)     │
 │  • Audio Stream: Cough / Wheeze / Respiratory Strain     │
 │  • Text Inputs: Symptom Logs & Wong-Baker Pain Scale    │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │              ORCHESTRATION & INFERENCE ENGINE            │
 │  • Multimodal Fusion (Correlate Vitals + Symptoms)       │
 │  • Deterministic Triage Safeguards (ESI Version 4)       │
 │  • Local Medical Brain (Ollama / Expert Clinical Engine) │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │                   OUTPUT INTERFACE LAYER                │
 │  • Animated Baymax Face Avatar (Blinking & Expressions)  │
 │  • Empathic Voice Synthesis (Pace & Pitch Modulation)    │
 │  • Real-Time PPG Waveform Dashboard & Escalation Dialog  │
 └─────────────────────────────────────────────────────────┘
```

---

## Key Modules & Technologies

### 1. Non-Contact Perception Layer
* **Photoplethysmography (rPPG)**: Implements the **Plane-Orthogonal-to-Skin (POS)** and **Chrominance (CHROM)** algorithms using OpenCV, MediaPipe, NumPy, and SciPy. Automatically extracts facial regions of interest (forehead and cheeks), applies 4th-order Butterworth bandpass filtering (0.7 Hz – 3.5 Hz), and computes Welch Power Spectral Density (PSD) to detect real-time **Heart Rate (BPM)** and **Heart Rate Variability (HRV / RMSSD)**.
* **Acoustic Diagnostic Classifiers**: Real-time microphone audio processing via FFT, Mel-scale energy distribution, and spectral centroid metrics. Classifies dry vs. productive coughs, wheezing (bronchial constriction), inspiratory stridor, and respiratory strain / dyspnea.
* **Affective Emotion Engine**: Analyzes voice prosody, pitch ($F_0$), period jitter, and amplitude shimmer to assess patient anxiety, stress, and pain vocalization levels.

### 2. Orchestration & Medical Intelligence
* **Deterministic Emergency Severity Index (ESI v4)**: A rule-based clinical safety engine that prevents LLM hallucinations from under-triaging emergencies:
  * **Level 1 (Resuscitation)**: Cardiac arrest, apnea, profound bradycardia.
  * **Level 2 (Emergent)**: Acute chest pain, stroke signs, inspiratory stridor, severe pain $\ge 8/10$ with vitals instability.
  * **Level 3 (Urgent)**: Persistent fever with productive cough, moderate abdominal pain.
  * **Level 4 (Less Urgent)**: Mild localized aches, simple sprains.
  * **Level 5 (Non-urgent)**: Routine wellness inquiries, self-care guidance.
* **Local Medical Diagnostic Brain**: Dual-layer architecture connecting to local **Ollama** (e.g., `BioMistral-7B`, `MedLlama-3`), an integrated deterministic Clinical Expert Knowledge Engine, and optional Google Gemini / OpenAI cloud fallback.

### 3. Empathic Baymax Interface
* **Animated Baymax Avatar**: Realistic human-robot ocular blinking, interactive pupil tracking, speaking ripples, and emotional expression states (`calm`, `listening`, `concerned`, `emergency`).
* **Affective Voice Engine**: Web Speech Synthesis and backend `pyttsx3` with dynamic pace, pitch, and soothing cadence modulation based on patient stress.
* **Real-Time Telemetry Dashboard**: 60 FPS arterial photoplethysmogram (PPG) pulse wave canvas, Wong-Baker FACES pain scale (0–10), and native emergency escalation modal with quick 911 dialer.

---

## Quick Start

### 1. Install Dependencies
Ensure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python run.py
```
This starts the FastAPI ASGI server on `http://127.0.0.1:8000` and automatically opens the cockpit interface in your browser.

### 3. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Serves the Baymax cyber-medical dashboard UI |
| `/api/status` | `GET` | Health check & loaded perception modules |
| `/api/vitals` | `GET` | Current fused optical and acoustic vitals |
| `/api/triage` | `POST` | Executes deterministic ESI v4 clinical triage |
| `/api/chat` | `POST` | Conversational clinical consultation with Baymax |
| `/api/synthesize`| `POST` | Generates modulated speech audio file |
| `/ws/telemetry` | `WebSocket` | Bi-directional live video/audio frame & vitals stream |

---

## Clinical Safety Disclaimer
*Baymax is a software health companion and educational demonstration. It does not replace professional medical advice, diagnosis, or treatment. In life-threatening emergencies, call 911 or visit your nearest emergency room immediately.*
