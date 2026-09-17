/**
 * Audio Stream & Acoustic Diagnostic Visualizer.
 * Captures microphone audio, computes frequency spectrum bars,
 * and transmits audio chunks to backend acoustic classifiers.
 */

class AudioStreamManager {
  constructor() {
    this.audioCtx = null;
    this.analyser = null;
    this.source = null;
    this.processor = null;
    this.stream = null;
    this.isActive = false;

    this.spectrumBars = [];
    this._initBars();
  }

  _initBars() {
    const container = document.getElementById('spectrum-bars');
    if (!container) return;
    container.innerHTML = '';
    this.spectrumBars = [];

    for (let i = 0; i < 16; i++) {
      const bar = document.createElement('div');
      bar.className = 'spectrum-bar';
      container.appendChild(bar);
      this.spectrumBars.push(bar);
    }
  }

  async startAudio(onAudioChunkCallback) {
    if (this.isActive) return;

    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
        },
        video: false,
      });

      this.source = this.audioCtx.createMediaStreamSource(this.stream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 512;

      // ScriptProcessor for chunk extraction (buffer size 4096 = 256ms @ 16kHz)
      this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (e) => {
        if (!this.isActive) return;
        const inputData = e.inputBuffer.getChannelData(0);
        // Copy to float array
        const samples = Array.from(inputData);

        if (onAudioChunkCallback) {
          onAudioChunkCallback(samples, this.audioCtx.sampleRate);
        }
      };

      this.source.connect(this.analyser);
      this.analyser.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);

      this.isActive = true;
      this._startVisualizerLoop();
      return true;
    } catch (err) {
      console.warn('Microphone permission denied or unavailable:', err);
      return false;
    }
  }

  _startVisualizerLoop() {
    if (!this.analyser) return;
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

    const render = () => {
      if (!this.isActive) return;

      this.analyser.getByteFrequencyData(dataArray);

      // Map to 16 bars
      const step = Math.floor(dataArray.length / 16);
      for (let i = 0; i < 16; i++) {
        let sum = 0;
        for (let j = 0; j < step; j++) {
          sum += dataArray[i * step + j];
        }
        const avg = sum / step;
        const pct = Math.max(5, Math.min(100, (avg / 255) * 100));

        if (this.spectrumBars[i]) {
          this.spectrumBars[i].style.height = `${pct}%`;
        }
      }

      requestAnimationFrame(render);
    };

    render();
  }

  updateAcousticFlags(acousticData) {
    const coughFlag = document.getElementById('flag-cough');
    const wheezeFlag = document.getElementById('flag-wheeze');
    const breathFlag = document.getElementById('flag-breath');

    if (coughFlag) {
      if (acousticData.is_cough) {
        coughFlag.className = 'diag-flag alert';
        coughFlag.innerHTML = `<span>COUGH</span> <strong>${acousticData.cough_type}</strong>`;
      } else {
        coughFlag.className = 'diag-flag';
        coughFlag.innerHTML = '<span>COUGH</span> <span>None</span>';
      }
    }

    if (wheezeFlag) {
      if (acousticData.wheeze_detected) {
        wheezeFlag.className = 'diag-flag alert';
        wheezeFlag.innerHTML = '<span>WHEEZE</span> <strong>ACTIVE</strong>';
      } else {
        wheezeFlag.className = 'diag-flag';
        wheezeFlag.innerHTML = '<span>WHEEZE</span> <span>Normal</span>';
      }
    }

    if (breathFlag) {
      if (acousticData.stridor_detected || acousticData.dyspnea_index > 0.5) {
        breathFlag.className = 'diag-flag alert';
        breathFlag.innerHTML = `<span>RESP STRAIN</span> <strong>${Math.round(acousticData.dyspnea_index * 100)}%</strong>`;
      } else {
        breathFlag.className = 'diag-flag';
        breathFlag.innerHTML = `<span>RESP RATE</span> <span>${acousticData.respiratory_rate} /m</span>`;
      }
    }
  }

  stopAudio() {
    this.isActive = false;
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
  }
}

window.audioStreamManager = new AudioStreamManager();
