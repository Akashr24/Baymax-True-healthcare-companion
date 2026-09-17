/**
 * Real-Time Telemetry Dashboard & Photoplethysmogram (PPG) Waveform Canvas Renderer.
 */

class DashboardManager {
  constructor(canvasId = 'ppg-canvas') {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

    this.wavePoints = [];
    this.maxPoints = 120;
    this.currentPulseVal = 0;
    this.animFrameId = null;

    this._initCanvasSize();
    this._startWaveAnimation();
  }

  _initCanvasSize() {
    if (!this.canvas) return;
    const resize = () => {
      this.canvas.width = this.canvas.clientWidth || 500;
      this.canvas.height = this.canvas.clientHeight || 110;
    };
    resize();
    window.addEventListener('resize', resize);
  }

  appendPulseValue(val) {
    this.currentPulseVal = val;
    this.wavePoints.push(val);
    if (this.wavePoints.length > this.maxPoints) {
      this.wavePoints.shift();
    }
  }

  _startWaveAnimation() {
    if (!this.ctx || !this.canvas) return;

    let time = 0;

    const render = () => {
      time += 0.04;
      const w = this.canvas.width;
      const h = this.canvas.height;
      const cy = h / 2;

      this.ctx.clearRect(0, 0, w, h);

      // If we don't have enough live points yet, generate a realistic resting PPG pulse wave
      let points = this.wavePoints;
      if (points.length < 10) {
        points = [];
        for (let i = 0; i < this.maxPoints; i++) {
          const t = time + i * 0.1;
          // Synthetic dicrotic notch cardiac pulse formula
          const primary = Math.sin(t * 2.8);
          const dicrotic = 0.35 * Math.sin(t * 5.6 + 1.2);
          points.push(primary + dicrotic);
        }
      }

      // Draw glowing pulse wave
      this.ctx.beginPath();
      const step = w / (points.length - 1 || 1);

      for (let i = 0; i < points.length; i++) {
        const x = i * step;
        // Normalize amplitude to canvas height
        const val = points[i];
        const y = cy - val * (h * 0.32);

        if (i === 0) {
          this.ctx.moveTo(x, y);
        } else {
          // Smooth bezier curve
          const prevX = (i - 1) * step;
          const prevVal = points[i - 1];
          const prevY = cy - prevVal * (h * 0.32);
          const cx = (prevX + x) / 2;
          this.ctx.quadraticCurveTo(prevX, prevY, cx, (prevY + y) / 2);
        }
      }

      // Stroke style
      this.ctx.strokeStyle = '#00f0ff';
      this.ctx.lineWidth = 2.5;
      this.ctx.shadowColor = '#00f0ff';
      this.ctx.shadowBlur = 10;
      this.ctx.stroke();
      this.ctx.shadowBlur = 0; // reset

      // Draw leading current blip marker
      if (points.length > 0) {
        const lastX = (points.length - 1) * step;
        const lastY = cy - points[points.length - 1] * (h * 0.32);

        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.strokeStyle = '#00f0ff';
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
        this.ctx.stroke();
      }

      this.animFrameId = requestAnimationFrame(render);
    };

    render();
  }

  updateVitals(telemetry) {
    const rppg = telemetry.rppg || {};
    const acoustic = telemetry.acoustic || {};
    const emotion = telemetry.emotion || {};
    const fusion = telemetry.fusion || {};
    const triage = telemetry.triage || {};

    // 1. Heart Rate
    const hrVal = document.getElementById('val-bpm');
    const hrSub = document.getElementById('status-bpm');
    if (hrVal && rppg.bpm) {
      hrVal.textContent = Math.round(rppg.bpm);
    }
    if (hrSub && fusion.hemodynamic_status) {
      hrSub.textContent = fusion.hemodynamic_status;
      if (rppg.bpm > 115 || rppg.bpm < 50) {
        hrSub.style.color = '#ff3366';
      } else {
        hrSub.style.color = '#10b981';
      }
    }

    // 2. HRV (RMSSD)
    const hrvVal = document.getElementById('val-hrv');
    const hrvSub = document.getElementById('status-hrv');
    if (hrvVal && rppg.hrv) {
      hrvVal.textContent = Math.round(rppg.hrv);
    }
    if (hrvSub && fusion.autonomic_state) {
      hrSub.textContent = fusion.autonomic_state;
    }

    // 3. Respiratory Rate
    const rrVal = document.getElementById('val-rr');
    const rrSub = document.getElementById('status-rr');
    if (rrVal && acoustic.respiratory_rate) {
      rrVal.textContent = Math.round(acoustic.respiratory_rate);
    }
    if (rrSub && fusion.pulmonary_status) {
      rrSub.textContent = fusion.pulmonary_status;
      if (acoustic.wheeze_detected || acoustic.stridor_detected) {
        rrSub.style.color = '#ff3366';
      } else {
        rrSub.style.color = '#10b981';
      }
    }

    // 4. Stress & Health Index
    const stressVal = document.getElementById('val-stress');
    const stressSub = document.getElementById('status-stress');
    if (stressVal && emotion.stress_level !== undefined) {
      stressVal.textContent = `${emotion.stress_level}%`;
    }
    if (stressSub && emotion.emotional_tone) {
      stressSub.textContent = emotion.emotional_tone;
    }

    // 5. Triage Verdict Card
    const triageBadge = document.getElementById('triage-badge');
    const triageActions = document.getElementById('triage-actions');

    if (triageBadge && triage.severity_label) {
      triageBadge.textContent = triage.severity_label;
      triageBadge.style.backgroundColor = `${triage.urgency_color}25`;
      triageBadge.style.color = triage.urgency_color;
      triageBadge.style.borderColor = triage.urgency_color;
    }

    if (triageActions && triage.immediate_actions) {
      triageActions.innerHTML = triage.immediate_actions
        .map((a) => `<li>${a}</li>`)
        .join('');
    }

    // Append latest pulse value to wave
    if (rppg.pulse_value !== undefined) {
      this.appendPulseValue(rppg.pulse_value);
    }
  }
}

window.dashboardManager = new DashboardManager();
