"""
rPPG (Remote Photoplethysmography) Engine.
Extracts cardiac pulse signals (Heart Rate & HRV) from facial video streams using
ROI localization and Plane-Orthogonal-to-Skin (POS) and Chrominance (CHROM) algorithms.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy import signal

# Optional MediaPipe face mesh / detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_detection = mp.solutions.face_detection
except Exception:
    MP_AVAILABLE = False


class RPPGEngine:
    """
    Non-contact cardiac pulse extraction engine.
    Processes sequential video frames, isolates facial capillary beds,
    and extracts Heart Rate (BPM) and HRV metrics via temporal color analysis.
    """

    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 6.0,
        min_bpm: float = 42.0,
        max_bpm: float = 210.0,
    ):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(self.fps * self.buffer_seconds)
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

        # Signal buffers
        self.r_buffer = deque(maxlen=self.buffer_size)
        self.g_buffer = deque(maxlen=self.buffer_size)
        self.b_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)

        # Pulse history for UI visualization
        self.pulse_history = deque(maxlen=100)

        # Face detector setup
        self._init_face_detector()

        # State cache
        self.last_bpm = 72.0
        self.last_hrv = 45.0  # RMSSD in ms
        self.last_confidence = 0.0
        self.last_face_rect: Optional[Dict[str, int]] = None
        self.last_rois: List[Dict[str, int]] = []
        self.frame_counter = 0

    def _init_face_detector(self):
        """Initializes MediaPipe face detector with Haar Cascade fallback."""
        self.mp_detector = None
        if MP_AVAILABLE:
            try:
                self.mp_detector = mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception:
                self.mp_detector = None

        # Load OpenCV Haar Cascade as guaranteed fallback
        self.haar_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.haar_cascade = None

    def detect_face_and_rois(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Locates the face bounding box and extracts key facial skin ROIs:
        Forehead, Left Malar (Cheek), Right Malar (Cheek).
        Returns: (face_box, [roi_forehead, roi_left_cheek, roi_right_cheek])
        """
        h, w = frame.shape[:2]
        face_box: Optional[Tuple[int, int, int, int]] = None

        # 1. Try MediaPipe
        if self.mp_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_detector.process(rgb_frame)
                if results.detections:
                    det = results.detections[0]
                    bb = det.location_data.relative_bounding_box
                    fx = int(max(0, bb.xmin * w))
                    fy = int(max(0, bb.ymin * h))
                    fw = int(min(w - fx, bb.width * w))
                    fh = int(min(h - fy, bb.height * h))
                    if fw > 40 and fh > 40:
                        face_box = (fx, fy, fw, fh)
            except Exception:
                face_box = None

        # 2. Fallback to Haar Cascade
        if face_box is None and self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_box = (int(fx), int(fy), int(fw), int(fh))

        # 3. Fallback: Center frame region if no detector detected
        if face_box is None:
            cw, ch = int(w * 0.4), int(h * 0.5)
            cx, cy = int(w * 0.3), int(h * 0.25)
            face_box = (cx, cy, cw, ch)

        fx, fy, fw, fh = face_box

        # Compute ROIs (Forehead, Left Cheek, Right Cheek)
        # Forehead: top 10%-25% of face, center 50% width
        forehead = (
            int(fx + fw * 0.25),
            int(fy + fh * 0.08),
            int(fw * 0.50),
            int(fh * 0.18),
        )

        # Left Cheek: 50%-75% height, 15%-40% width
        left_cheek = (
            int(fx + fw * 0.15),
            int(fy + fh * 0.50),
            int(fw * 0.25),
            int(fh * 0.22),
        )

        # Right Cheek: 50%-75% height, 60%-85% width
        right_cheek = (
            int(fx + fw * 0.60),
            int(fy + fh * 0.50),
            int(fw * 0.25),
            int(fh * 0.22),
        )

        rois = [forehead, left_cheek, right_cheek]
        return face_box, rois

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Ingests a single video frame, computes the spatial RGB mean across skin ROIs,
        updates temporal buffers, and calculates heart rate and HRV if sufficient buffer is present.
        """
        self.frame_counter += 1
        curr_time = time.time()
        h, w = frame.shape[:2]

        face_box, rois = self.detect_face_and_rois(frame)

        # Store for serialization
        if face_box:
            self.last_face_rect = {
                "x": face_box[0],
                "y": face_box[1],
                "w": face_box[2],
                "h": face_box[3],
            }
        self.last_rois = [
            {"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for r in rois
        ]

        # Extract spatial mean for RGB channels
        roi_means = []
        for rx, ry, rw, rh in rois:
            # Ensure boundaries are inside frame
            x1 = max(0, min(w - 1, rx))
            y1 = max(0, min(h - 1, ry))
            x2 = max(0, min(w, rx + rw))
            y2 = max(0, min(h, ry + rh))

            if x2 > x1 and y2 > y1:
                patch = frame[y1:y2, x1:x2]
                # OpenCV uses BGR
                mean_b = float(np.mean(patch[:, :, 0]))
                mean_g = float(np.mean(patch[:, :, 1]))
                mean_r = float(np.mean(patch[:, :, 2]))
                roi_means.append((mean_r, mean_g, mean_b))

        if roi_means:
            avg_r = float(np.mean([m[0] for m in roi_means]))
            avg_g = float(np.mean([m[1] for m in roi_means]))
            avg_b = float(np.mean([m[2] for m in roi_means]))
        else:
            avg_r, avg_g, avg_b = 128.0, 128.0, 128.0

        # Append to temporal queues
        self.r_buffer.append(avg_r)
        self.g_buffer.append(avg_g)
        self.b_buffer.append(avg_b)
        self.timestamps.append(curr_time)

        # Calculate actual FPS from timestamps if possible
        if len(self.timestamps) >= 30:
            dt = self.timestamps[-1] - self.timestamps[0]
            if dt > 0.5:
                self.fps = (len(self.timestamps) - 1) / dt

        # Need at least 2.5 seconds to start estimating pulse
        min_required = int(self.fps * 2.5)
        if len(self.g_buffer) >= min_required:
            bpm, hrv, conf, pulse_val = self._compute_vitals_pos()
            self.last_bpm = bpm
            self.last_hrv = hrv
            self.last_confidence = conf
            self.pulse_history.append(float(pulse_val))
        else:
            pulse_val = 0.0
            self.pulse_history.append(0.0)

        return {
            "bpm": round(self.last_bpm, 1),
            "hrv": round(self.last_hrv, 1),
            "confidence": round(self.last_confidence, 2),
            "pulse_value": round(float(pulse_val), 4),
            "pulse_wave": list(self.pulse_history),
            "face_detected": self.last_face_rect is not None,
            "face_rect": self.last_face_rect,
            "rois": self.last_rois,
            "buffer_progress": min(1.0, len(self.g_buffer) / float(self.buffer_size)),
        }

    def _compute_vitals_pos(self) -> Tuple[float, float, float, float]:
        """
        Plane-Orthogonal-to-Skin (POS) algorithm + Welch Spectral Analysis.
        Returns: (BPM, HRV_RMSSD_ms, confidence, latest_pulse_value)
        """
        r = np.array(self.r_buffer, dtype=np.float64)
        g = np.array(self.g_buffer, dtype=np.float64)
        b = np.array(self.b_buffer, dtype=np.float64)
        n = len(r)

        # 1. Temporal mean normalization
        mean_r = np.mean(r) or 1.0
        mean_g = np.mean(g) or 1.0
        mean_b = np.mean(b) or 1.0

        rn = r / mean_r
        gn = g / mean_g
        bn = b / mean_b

        # 2. Chrominance projection vectors
        # S1 = G - B
        # S2 = G + B - 2R
        s1 = gn - bn
        s2 = gn + bn - 2.0 * rn

        std_s2 = np.std(s2)
        if std_s2 < 1e-6:
            alpha = 1.0
        else:
            alpha = np.std(s1) / std_s2

        # 3. Pulse signal H
        h_signal = s1 + alpha * s2

        # 4. Bandpass filter (Butterworth 4th order: 0.7 Hz to 3.5 Hz)
        lowcut = max(0.6, self.min_bpm / 60.0)
        highcut = min(self.fps / 2.0 - 0.1, self.max_bpm / 60.0)
        nyq = 0.5 * self.fps

        if highcut <= lowcut or nyq <= highcut:
            # Fallback for small fps
            return self.last_bpm, self.last_hrv, 0.2, 0.0

        b_filter, a_filter = signal.butter(
            4, [lowcut / nyq, highcut / nyq], btype="bandpass"
        )
        try:
            # Detrend first
            h_detrended = signal.detrend(h_signal)
            filtered = signal.filtfilt(b_filter, a_filter, h_detrended)
        except Exception:
            filtered = h_signal - np.mean(h_signal)

        latest_pulse = float(filtered[-1])

        # 5. Welch's Power Spectral Density (PSD) for Dominant BPM Frequency
        nperseg = min(len(filtered), int(self.fps * 4.0))
        if nperseg < 16:
            nperseg = len(filtered)

        freqs, psd = signal.welch(filtered, fs=self.fps, nperseg=nperseg)

        # Restrict to physiological range [0.7 Hz, 3.5 Hz]
        valid_mask = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(valid_mask):
            return self.last_bpm, self.last_hrv, 0.3, latest_pulse

        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]

        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        detected_bpm = float(peak_freq * 60.0)

        # Confidence metric based on peak prominence over background noise
        total_power = np.sum(valid_psd) + 1e-9
        peak_power = valid_psd[peak_idx]
        confidence = float(np.clip((peak_power / total_power) * 3.5, 0.1, 0.98))

        # 6. Peak detection for Inter-Beat Interval (IBI) and HRV (RMSSD)
        # Distance between peaks minimum based on max_bpm
        min_peak_distance = int(self.fps * (60.0 / self.max_bpm))
        peaks, _ = signal.find_peaks(
            filtered,
            distance=max(1, min_peak_distance),
            prominence=0.2 * np.std(filtered),
        )

        hrv_rmssd = self.last_hrv
        if len(peaks) >= 3:
            # Convert peak sample indices to time intervals (ms)
            peak_times_ms = (peaks / self.fps) * 1000.0
            ibis = np.diff(peak_times_ms)
            # Filter physiologically plausible intervals (300ms to 1400ms)
            valid_ibis = ibis[(ibis >= 300.0) & (ibis <= 1400.0)]
            if len(valid_ibis) >= 2:
                successive_diffs = np.diff(valid_ibis)
                hrv_rmssd = float(np.sqrt(np.mean(successive_diffs**2)))
                # Constrain HRV to clinical range 10ms - 150ms
                hrv_rmssd = float(np.clip(hrv_rmssd, 12.0, 135.0))

        # Temporal smoothing with previous estimate
        smooth_bpm = 0.7 * self.last_bpm + 0.3 * detected_bpm
        smooth_hrv = 0.8 * self.last_hrv + 0.2 * hrv_rmssd

        return smooth_bpm, smooth_hrv, confidence, latest_pulse
