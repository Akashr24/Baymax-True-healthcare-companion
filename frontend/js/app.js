/**
 * Baymax Healthcare Companion - Master Application Orchestrator.
 * Connects WebSocket telemetry, handles pain scale interactions,
 * coordinates triage escalations, and manages conversational consultations.
 */

class BaymaxApp {
  constructor() {
    this.ws = null;
    this.wsConnected = false;
    this.currentPainScore = 0;
    this.currentSymptoms = '';
    this.isEmergencyModalOpen = false;

    this._initElements();
    this._initWebSocket();
    this._initPainScale();
    this._initSymptomChips();
    this._initControls();
  }

  _initElements() {
    this.quoteText = document.getElementById('baymax-quote-text');
    this.quoteSource = document.getElementById('quote-source');
    this.symptomInput = document.getElementById('symptom-input');
    this.consultBtn = document.getElementById('consult-btn');
    this.emergencyDialog = document.getElementById('emergency-dialog');
    this.dismissEmergencyBtn = document.getElementById('dismiss-emergency');
    this.emergencyHeaderBtn = document.getElementById('btn-emergency-sos');

    this.toggleCamBtn = document.getElementById('btn-toggle-cam');
    this.toggleMicBtn = document.getElementById('btn-toggle-mic');
    this.toggleVoiceBtn = document.getElementById('btn-toggle-voice');
    this.connectionDot = document.getElementById('conn-status-dot');
    this.connectionText = document.getElementById('conn-status-text');

    if (this.consultBtn) {
      this.consultBtn.addEventListener('click', () => this.handleConsultation());
    }

    if (this.symptomInput) {
      this.symptomInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          this.handleConsultation();
        }
      });
    }

    if (this.emergencyHeaderBtn) {
      this.emergencyHeaderBtn.addEventListener('click', () => {
        this.openEmergencyModal({
          severity_label: 'EMERGENCY OVERRIDE (User Triggered SOS)',
          immediate_actions: [
            'CALL 911 OR LOCAL EMERGENCY DISPATCH IMMEDIATELY.',
            'Sit in a secure upright posture. Unlock primary exit door.',
            'Keep calm; Baymax is maintaining physiological telemetry monitoring.',
          ],
        });
      });
    }

    if (this.dismissEmergencyBtn && this.emergencyDialog) {
      this.dismissEmergencyBtn.addEventListener('click', () => {
        this.emergencyDialog.close();
        this.isEmergencyModalOpen = false;
        if (window.baymaxAvatar) {
          window.baymaxAvatar.setExpression('calm');
        }
      });
    }
  }

  _initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const wsUrl = `${protocol}//${host}/ws/telemetry`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.wsConnected = true;
        if (this.connectionDot) this.connectionDot.className = 'status-dot';
        if (this.connectionText) this.connectionText.textContent = 'TELEMETRY ONLINE';
        console.log('Baymax WebSocket Telemetry connected.');
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'telemetry') {
            this.handleTelemetryUpdate(msg);
          }
        } catch (err) {
          console.error('Error parsing telemetry packet:', err);
        }
      };

      this.ws.onclose = () => {
        this.wsConnected = false;
        if (this.connectionDot) this.connectionDot.className = 'status-dot active-red';
        if (this.connectionText) this.connectionText.textContent = 'RECONNECTING...';
        setTimeout(() => this._initWebSocket(), 3000);
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket error:', err);
      };
    } catch (e) {
      console.warn('Could not connect to WebSocket:', e);
    }
  }

  sendWsMessage(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  handleTelemetryUpdate(telemetry) {
    // 1. Update dashboard charts and values
    if (window.dashboardManager) {
      window.dashboardManager.updateVitals(telemetry);
    }

    // 2. Update face bounding box overlays
    if (window.cameraStreamManager && telemetry.rppg) {
      window.cameraStreamManager.updateFaceTelemetry(
        telemetry.rppg.face_rect,
        telemetry.rppg.rois
      );
    }

    // 3. Update acoustic flags
    if (window.audioStreamManager && telemetry.acoustic) {
      window.audioStreamManager.updateAcousticFlags(telemetry.acoustic);
    }

    // 4. Check for Emergency Triage Escalation
    const triage = telemetry.triage || {};
    if (triage.is_emergency && !this.isEmergencyModalOpen) {
      this.openEmergencyModal(triage);
    }
  }

  _initPainScale() {
    const painButtons = document.querySelectorAll('.pain-btn');
    const scorePill = document.getElementById('pain-score-pill');

    painButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        painButtons.forEach((b) => b.classList.remove('selected'));
        btn.classList.add('selected');

        const score = parseInt(btn.getAttribute('data-score'), 10) || 0;
        this.currentPainScore = score;
        if (scorePill) scorePill.textContent = `${score} / 10`;

        // Transmit update to backend
        this.sendWsMessage({
          type: 'patient_update',
          pain_score: score,
          symptoms: this.currentSymptoms,
        });

        // Expression reaction from Baymax
        if (window.baymaxAvatar) {
          if (score >= 8) {
            window.baymaxAvatar.setExpression('emergency');
          } else if (score >= 4) {
            window.baymaxAvatar.setExpression('concerned');
          } else {
            window.baymaxAvatar.setExpression('calm');
          }
        }
      });
    });
  }

  _initSymptomChips() {
    const chips = document.querySelectorAll('.symptom-chip');
    chips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const text = chip.getAttribute('data-symptom') || chip.textContent;
        if (this.symptomInput) {
          this.symptomInput.value = text;
          this.handleConsultation();
        }
      });
    });
  }

  _initControls() {
    // Camera toggle
    if (this.toggleCamBtn) {
      this.toggleCamBtn.addEventListener('click', async () => {
        if (!window.cameraStreamManager.isActive) {
          const started = await window.cameraStreamManager.startCamera((dataUrl) => {
            this.sendWsMessage({ type: 'video_frame', image: dataUrl });
          });
          if (started) {
            this.toggleCamBtn.classList.add('active');
            this.toggleCamBtn.innerHTML = `
              <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
              Camera Active
            `;
          }
        } else {
          window.cameraStreamManager.stopCamera();
          this.toggleCamBtn.classList.remove('active');
          this.toggleCamBtn.innerHTML = `
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            Start Camera
          `;
        }
      });
    }

    // Microphone toggle
    if (this.toggleMicBtn) {
      this.toggleMicBtn.addEventListener('click', async () => {
        if (!window.audioStreamManager.isActive) {
          const started = await window.audioStreamManager.startAudio((samples, sr) => {
            this.sendWsMessage({ type: 'audio_chunk', samples: samples, sample_rate: sr });
          });
          if (started) {
            this.toggleMicBtn.classList.add('active');
            this.toggleMicBtn.innerHTML = `
              <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
              Acoustics Active
            `;
          }
        } else {
          window.audioStreamManager.stopAudio();
          this.toggleMicBtn.classList.remove('active');
          this.toggleMicBtn.innerHTML = `
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
            Start Acoustics
          `;
        }
      });
    }

    // Voice toggle
    if (this.toggleVoiceBtn) {
      this.toggleVoiceBtn.addEventListener('click', () => {
        const isEnabled = window.baymaxVoiceSynth.toggleVoice();
        if (isEnabled) {
          this.toggleVoiceBtn.classList.add('active');
          this.toggleVoiceBtn.textContent = 'Voice: ON';
        } else {
          this.toggleVoiceBtn.classList.remove('active');
          this.toggleVoiceBtn.textContent = 'Voice: MUTED';
        }
      });
    }

    // Auto-prompt initial greeting after short delay
    setTimeout(() => {
      const initialGreeting =
        "Hello, I am Baymax, your personal healthcare companion. I was alerted to the need for medical attention. On a scale of 1 to 10, how would you rate your pain?";
      if (this.quoteText) this.quoteText.textContent = initialGreeting;
      if (this.quoteSource) this.quoteSource.textContent = "Baymax Healthcare Protocol v1.0";
      if (window.baymaxVoiceSynth) {
        window.baymaxVoiceSynth.speak(initialGreeting, { rate: 0.90, pitch: 0.92 });
      }
    }, 1000);
  }

  async handleConsultation() {
    if (!this.symptomInput) return;
    const text = this.symptomInput.value.trim();
    if (!text) return;

    this.currentSymptoms = text;

    if (this.quoteText) {
      this.quoteText.textContent = "Scanning physiological indicators and conducting triage evaluation...";
    }
    if (window.baymaxAvatar) {
      window.baymaxAvatar.setExpression('listening');
      window.baymaxAvatar.triggerBlink();
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          pain_score: this.currentPainScore,
          symptoms: text,
        }),
      });

      if (!res.ok) throw new Error('API consultation request failed');
      const data = await res.json();

      if (this.quoteText) {
        this.quoteText.textContent = data.reply;
      }
      if (this.quoteSource) {
        this.quoteSource.textContent = `Diagnostic Source: ${data.source} | ESI Level ${data.triage.esi_level}`;
      }

      // Voice synthesis with Baymax affective parameters
      if (window.baymaxVoiceSynth) {
        window.baymaxVoiceSynth.speak(data.reply, data.speech_modulation || {});
      }

      // Check emergency escalation
      if (data.triage && data.triage.is_emergency) {
        this.openEmergencyModal(data.triage);
      } else if (this.currentPainScore >= 5) {
        if (window.baymaxAvatar) window.baymaxAvatar.setExpression('concerned');
      } else {
        if (window.baymaxAvatar) window.baymaxAvatar.setExpression('calm');
      }
    } catch (err) {
      console.error('Consultation error:', err);
      if (this.quoteText) {
        this.quoteText.textContent =
          "I am having difficulty connecting to my central reasoning unit, but my local vital sensors remain active. Please rest in a comfortable position and stay calm.";
      }
    }
  }

  openEmergencyModal(triageData) {
    this.isEmergencyModalOpen = true;
    if (window.baymaxAvatar) {
      window.baymaxAvatar.setExpression('emergency');
    }

    const titleElem = document.getElementById('emergency-title');
    const calloutsElem = document.getElementById('emergency-callouts');

    if (titleElem) {
      titleElem.textContent = triageData.severity_label || 'CRITICAL MEDICAL ALERT';
    }

    if (calloutsElem && triageData.immediate_actions) {
      calloutsElem.innerHTML = triageData.immediate_actions
        .map(
          (act) => `
          <div class="callout-item">
            <svg width="18" height="18" fill="none" stroke="#ff3366" stroke-width="2" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            <span>${act}</span>
          </div>
        `
        )
        .join('');
    }

    if (this.emergencyDialog && typeof this.emergencyDialog.showModal === 'function') {
      this.emergencyDialog.showModal();
    }
  }
}

// Instantiate on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.baymaxApp = new BaymaxApp();
});
