/**
 * Camera Stream & rPPG Frame Transmitter.
 * Manages webcam media stream, canvas downsampling, and face ROI overlay rendering.
 */

class CameraStreamManager {
  constructor(videoElementId = 'webcam-video', canvasElementId = 'overlay-canvas') {
    this.video = document.getElementById(videoElementId);
    this.canvas = document.getElementById(canvasElementId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

    this.stream = null;
    this.isActive = false;
    this.frameInterval = null;
    this.targetFps = 20; // 20 FPS for bandwidth efficiency

    // Hidden offscreen canvas for frame downsampling
    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = 320;
    this.captureCanvas.height = 240;
    this.captureCtx = this.captureCanvas.getContext('2d');

    this.lastFaceRect = null;
    this.lastRois = [];
  }

  async startCamera(onFrameCallback) {
    if (this.isActive) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
      }

      this.isActive = true;

      // Adjust canvas resolution to match video
      if (this.canvas && this.video) {
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
      }

      // Start frame capture loop
      this.frameInterval = setInterval(() => {
        if (!this.isActive || !this.video || this.video.readyState < 2) return;

        // Draw downsampled frame to offscreen canvas
        this.captureCtx.drawImage(
          this.video,
          0,
          0,
          this.captureCanvas.width,
          this.captureCanvas.height
        );
        const dataUrl = this.captureCanvas.toDataURL('image/jpeg', 0.65);

        if (onFrameCallback) {
          onFrameCallback(dataUrl);
        }

        this.renderOverlays();
      }, 1000 / this.targetFps);

      return true;
    } catch (err) {
      console.warn('Webcam permission denied or unavailable:', err);
      // Fallback: Start simulated camera frames so user can test without webcam
      this._startSimulatedFeed(onFrameCallback);
      return false;
    }
  }

  _startSimulatedFeed(onFrameCallback) {
    this.isActive = true;
    let t = 0;

    this.frameInterval = setInterval(() => {
      t += 0.05;
      const pulseMod = 0.02 * Math.sin(2 * Math.PI * 1.25 * t); // 75 BPM

      // Draw synthetic face on capture canvas
      this.captureCtx.fillStyle = '#1e293b';
      this.captureCtx.fillRect(0, 0, 320, 240);

      // Face oval
      this.captureCtx.fillStyle = `rgb(${180}, ${Math.floor(140 + pulseMod * 100)}, ${130})`;
      this.captureCtx.beginPath();
      this.captureCtx.ellipse(160, 120, 60, 80, 0, 0, Math.PI * 2);
      this.captureCtx.fill();

      const dataUrl = this.captureCanvas.toDataURL('image/jpeg', 0.6);
      if (onFrameCallback) {
        onFrameCallback(dataUrl);
      }
    }, 1000 / this.targetFps);
  }

  stopCamera() {
    this.isActive = false;
    if (this.frameInterval) {
      clearInterval(this.frameInterval);
      this.frameInterval = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  updateFaceTelemetry(faceRect, rois) {
    this.lastFaceRect = faceRect;
    this.lastRois = rois || [];
  }

  renderOverlays() {
    if (!this.ctx || !this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Coordinate scale factor between capture (320x240) and display canvas
    const scaleX = this.canvas.width / (this.captureCanvas.width || 320);
    const scaleY = this.canvas.height / (this.captureCanvas.height || 240);

    // Draw main face bounding box with cyber-brackets
    if (this.lastFaceRect) {
      const fx = this.lastFaceRect.x * scaleX;
      const fy = this.lastFaceRect.y * scaleY;
      const fw = this.lastFaceRect.w * scaleX;
      const fh = this.lastFaceRect.h * scaleY;

      this.ctx.strokeStyle = '#00f0ff';
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(fx, fy, fw, fh);

      // HUD corner tags
      this.ctx.fillStyle = 'rgba(0, 240, 255, 0.8)';
      this.ctx.font = '10px "JetBrains Mono", monospace';
      this.ctx.fillText('FACIAL CAPILLARY ROI TRACKED', fx + 4, fy - 6);
    }

    // Draw localized skin micro-ROIs (forehead & cheeks)
    if (this.lastRois && this.lastRois.length > 0) {
      this.lastRois.forEach((roi, idx) => {
        const rx = roi.x * scaleX;
        const ry = roi.y * scaleY;
        const rw = roi.w * scaleX;
        const rh = roi.h * scaleY;

        this.ctx.strokeStyle = idx === 0 ? '#10b981' : '#38bdf8';
        this.ctx.lineWidth = 1.5;
        this.ctx.setLineDash([4, 2]);
        this.ctx.strokeRect(rx, ry, rw, rh);
        this.ctx.setLineDash([]);
      });
    }
  }
}

window.cameraStreamManager = new CameraStreamManager();
