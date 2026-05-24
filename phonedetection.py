"""
phonedetection.py
-----------------
Real-time attention monitoring for online lectures.

Detects three inattention signals from a webcam feed:
  1. Eye closure  — eye aspect ratio (EAR) below threshold for N frames
  2. Face absence — no face detected for N frames
  3. Phone use    — MobileNet/GoogLeNet detects a cellular telephone

An alarm sound is played (optional) and an on-screen alert is shown when
any signal is triggered.

Usage:
    python phonedetection.py \
        --shape-predictor assets/shape_predictor_68_face_landmarks.dat \
        --prototxt assets/bvlc_googlenet.prototxt \
        --model assets/bvlc_googlenet.caffemodel \
        --labels assets/synset_words.txt \
        [--alarm assets/alarm.wav]

Requirements:
    pip install opencv-python dlib imutils scipy pygame
"""

import argparse
import time
from threading import Thread

import cv2
import dlib
import imutils
import numpy as np
import pygame
from imutils import face_utils
from imutils.video import FPS, VideoStream
from scipy.spatial import distance as dist

# ── Thresholds & constants ─────────────────────────────────────
EYE_AR_THRESH = 0.3          # EAR below this → eye considered closed
CONSEC_FRAMES = 48           # Closed/missing frames before alarm fires
PHONE_CONSEC_FRAMES = 5      # Phone-detection frames before alarm fires
FRAME_WIDTH = 450            # Resize width for processing speed
TOP5_PREDICTIONS = 5         # Number of top predictions to check
PHONE_CLASS_LABEL = "cellular telephone"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time attention span detector.")
    parser.add_argument("-s", "--shape-predictor", required=True,
                        help="Path to dlib facial landmark predictor (.dat).")
    parser.add_argument("-a", "--alarm", type=str, default="",
                        help="Path to alarm WAV file (optional).")
    parser.add_argument("-p", "--prototxt", required=True,
                        help="Path to Caffe deploy prototxt file.")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to Caffe pre-trained model weights.")
    parser.add_argument("-l", "--labels", required=True,
                        help="Path to ImageNet synset_words.txt labels file.")
    return parser.parse_args()


def sound_alarm(path: str) -> None:
    """Play an alarm sound in a non-blocking way."""
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A low EAR indicates the eye is closing or closed.
    """
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def trigger_alarm(alarm_path: str, alarm_flag: bool) -> bool:
    """Fire an alarm once if not already active. Returns True (alarm is now on)."""
    if not alarm_flag and alarm_path:
        t = Thread(target=sound_alarm, args=(alarm_path,))
        t.daemon = True
        t.start()
    return True


def main() -> None:
    args = parse_args()
    pygame.init()

    # Load class labels
    rows = open(args.labels).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    # Load object-detection network
    print("[INFO] Loading object detection model...")
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    # Load facial landmark predictor
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # State counters
    blink_counter = 0
    face_absent_counter = 0
    phone_counter = 0
    alarm_drowsy = False
    alarm_face = False
    alarm_phone = False

    # Start video stream
    print("[INFO] Starting video stream...")
    vs = VideoStream(0).start()
    fps = FPS().start()
    time.sleep(1.0)

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            any_alarm_active = alarm_drowsy or alarm_face or alarm_phone

            # ── Face-absence detection ─────────────────────────
            if len(rects) == 0:
                face_absent_counter += 1
                if face_absent_counter >= CONSEC_FRAMES and not any_alarm_active:
                    print("[ALERT] No face detected")
                    alarm_face = trigger_alarm(args.alarm, alarm_face)
                cv2.putText(frame, "No Face Alert!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                face_absent_counter = 0
                alarm_face = False

            # ── Drowsiness (eye closure) detection ────────────
            if not face_absent_counter and not phone_counter:
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    left_eye = shape[l_start:l_end]
                    right_eye = shape[r_start:r_end]
                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                    # Draw eye contours
                    for eye in (left_eye, right_eye):
                        hull = cv2.convexHull(eye)
                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

                    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if ear < EYE_AR_THRESH:
                        blink_counter += 1
                        if blink_counter >= CONSEC_FRAMES and not any_alarm_active:
                            print("[ALERT] Drowsiness detected")
                            alarm_drowsy = trigger_alarm(args.alarm, alarm_drowsy)
                        cv2.putText(frame, "Drowsiness Alert!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        blink_counter = 0
                        alarm_drowsy = False

            # ── Phone detection ────────────────────────────────
            if not face_absent_counter and not blink_counter:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (224, 224)), 1, (224, 224), (104, 117, 123)
                )
                net.setInput(blob)
                predictions = net.forward()
                top_indices = np.argsort(predictions[0])[::-1][:TOP5_PREDICTIONS]

                phone_detected = any(
                    classes[idx] == PHONE_CLASS_LABEL for idx in top_indices
                )

                if phone_detected:
                    phone_counter += 1
                    print("[ALERT] Phone detected")
                else:
                    phone_counter = 0
                    alarm_phone = False

                if phone_counter >= PHONE_CONSEC_FRAMES and not any_alarm_active:
                    alarm_phone = trigger_alarm(args.alarm, alarm_phone)
                    cv2.putText(frame, "Cellphone Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                phone_counter = 0

            fps.update()
            cv2.imshow("Attention Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        fps.stop()
        print(f"[INFO] Elapsed: {fps.elapsed():.2f}s  FPS: {fps.fps():.2f}")
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    main()
