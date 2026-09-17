/**
 * Baymax Affective Speech Synthesis (Web Speech API + Backend Fallback).
 * Controls pitch, pace, and speech ripples synchronized with avatar face.
 */

class BaymaxVoiceSynth {
  constructor() {
    this.synth = window.speechSynthesis;
    this.enabled = true;
    this.preferredVoice = null;

    this._initVoices();
  }

  _initVoices() {
    if (!this.synth) return;

    const loadVoices = () => {
      const voices = this.synth.getVoices();
      // Search for calm, clear English voice
      this.preferredVoice =
        voices.find(
          (v) =>
            v.lang.startsWith('en') &&
            (v.name.includes('Natural') ||
              v.name.includes('Google') ||
              v.name.includes('David') ||
              v.name.includes('Samantha'))
        ) ||
        voices.find((v) => v.lang.startsWith('en')) ||
        voices[0];
    };

    loadVoices();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = loadVoices;
    }
  }

  speak(text, modulation = {}) {
    if (!this.synth || !this.enabled || !text) return;

    // Cancel any ongoing speech
    this.synth.cancel();

    // Strip markdown formatting characters for clean speech
    const cleanText = text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/•/g, '')
      .replace(/#/g, '')
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);

    if (this.preferredVoice) {
      utterance.voice = this.preferredVoice;
    }

    // Apply Baymax affective parameters
    utterance.rate = modulation.rate || 0.90;
    utterance.pitch = modulation.pitch || 0.92;
    utterance.volume = modulation.volume || 0.95;

    // Sync speech animation with Baymax avatar
    utterance.onstart = () => {
      if (window.baymaxAvatar) {
        window.baymaxAvatar.setSpeaking(true);
      }
    };

    utterance.onend = () => {
      if (window.baymaxAvatar) {
        window.baymaxAvatar.setSpeaking(false);
      }
    };

    utterance.onerror = () => {
      if (window.baymaxAvatar) {
        window.baymaxAvatar.setSpeaking(false);
      }
    };

    this.synth.speak(utterance);
  }

  toggleVoice() {
    this.enabled = !this.enabled;
    if (!this.enabled && this.synth) {
      this.synth.cancel();
      if (window.baymaxAvatar) {
        window.baymaxAvatar.setSpeaking(false);
      }
    }
    return this.enabled;
  }
}

window.baymaxVoiceSynth = new BaymaxVoiceSynth();
