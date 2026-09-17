/**
 * Baymax Interactive Face Avatar Controller.
 * Manages autonomous ocular blinking, pupil tracking, speech pulse ripples,
 * and empathetic expression states.
 */

class BaymaxAvatar {
  constructor(containerId = 'baymax-face') {
    this.container = document.getElementById(containerId);
    this.leftEye = document.getElementById('avatar-left-eye');
    this.rightEye = document.getElementById('avatar-right-eye');
    this.speechRing = document.getElementById('speaking-ring');

    this.currentState = 'calm';
    this.isSpeaking = false;
    this.blinkTimer = null;

    this._initBlinking();
    this._initPupilTracking();
  }

  _initBlinking() {
    // Spontaneous realistic human-robot blinking
    const scheduleNextBlink = () => {
      const delay = Math.random() * 3500 + 2500; // 2.5s - 6s
      this.blinkTimer = setTimeout(() => {
        this.triggerBlink();
        scheduleNextBlink();
      }, delay);
    };
    scheduleNextBlink();
  }

  triggerBlink() {
    if (!this.container) return;
    this.container.classList.add('blinking');
    setTimeout(() => {
      this.container.classList.remove('blinking');
    }, 160);
  }

  _initPupilTracking() {
    // Subtle ocular tracking of pointer cursor
    window.addEventListener('mousemove', (e) => {
      if (!this.container || this.currentState === 'emergency') return;

      const rect = this.container.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;

      const dx = (e.clientX - cx) / window.innerWidth;
      const dy = (e.clientY - cy) / window.innerHeight;

      const moveX = Math.max(-4, Math.min(4, dx * 10));
      const moveY = Math.max(-3, Math.min(3, dy * 8));

      if (this.leftEye) this.leftEye.style.transform = `translate(${moveX}px, ${moveY}px)`;
      if (this.rightEye) this.rightEye.style.transform = `translate(${moveX}px, ${moveY}px)`;
    });
  }

  setSpeaking(speaking) {
    this.isSpeaking = speaking;
    if (!this.container) return;

    if (speaking) {
      this.container.classList.add('is-speaking');
    } else {
      this.container.classList.remove('is-speaking');
    }
  }

  setExpression(state) {
    if (!this.container) return;
    this.currentState = state;

    this.container.classList.remove(
      'avatar-concerned',
      'avatar-emergency',
      'avatar-attentive'
    );

    if (state === 'concerned') {
      this.container.classList.add('avatar-concerned');
    } else if (state === 'emergency') {
      this.container.classList.add('avatar-emergency');
    } else if (state === 'listening') {
      this.container.classList.add('avatar-attentive');
    }
  }
}

// Export singleton instance to window
window.baymaxAvatar = new BaymaxAvatar();
