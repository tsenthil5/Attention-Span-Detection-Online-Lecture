# Attention Span Detection for Online Lectures

> Real-Time AI · EdTech · IEEE Published · Aug 2020 – Jul 2021

Real-time system that monitors and quantifies student attention during online instructor-led sessions using computer vision and audio processing. Delivers a live dashboard showing per-participant attention scores and fires automated alerts for disengagement.

**📄 IEEE Published** — Selected in the **top 10 of 100+ submissions** at the International Conference on Advances in Computing and Communication (ICACC 2021).  
👉 [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9299647)

## Highlights
- **85% attention-scoring accuracy** using a multi-signal fusion model
- Real-time visual cue processing via **Apache Kafka** streaming pipeline
- Live participation dashboard with per-learner attention scores
- Automatic trigger alerts for sub-optimal engagement
- Tracks face presence, eye contact, phone use, and dialogue ratio

## System Architecture
```
Video/Audio Stream
        │
   ┌────┴─────────────────┐
   │                       │
Phone Detection        Audio Analysis
(OpenCV + dlib)       (background noise,
   │                   dialogue ratio)
   │                       │
   └────────┬──────────────┘
            │
      Kafka Message Queue       ← real-time event stream
            │
     Attention Scoring Engine   ← weighted multi-signal fusion
            │
     Flask REST API
            │
     React Dashboard            ← live scores + alerts
```

## Results
| Metric | Value |
|---|---|
| Attention Scoring Accuracy | **85%** |
| Processing Latency | Real-time (<500ms) |
| Conference Recognition | Top 10 / 100+ papers, ICACC 2021 |

---

## Setup & Running

### 1. System Dependencies

**macOS**
```bash
brew install cmake portaudio
```

**Ubuntu/Debian**
```bash
sudo apt-get install cmake portaudio19-dev
```

> `cmake` is required to compile `dlib`. `portaudio` is required for `pyaudio` (audio.py).

---

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `dlib` will compile from source — this takes 3–5 minutes. Ensure `cmake` is installed first.

---

### 3. Download Model Assets

Create an `assets/` folder and download the following files into it:

```bash
mkdir assets
```

#### dlib facial landmark predictor
```bash
curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o assets/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 assets/shape_predictor_68_face_landmarks.dat.bz2
```

#### GoogLeNet Caffe model (phone detection)
```bash
# Model weights (~50MB)
curl -L "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" -o assets/bvlc_googlenet.caffemodel

# Deploy prototxt
curl -L "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt" -o assets/bvlc_googlenet.prototxt

# ImageNet class labels
curl -L "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt" -o assets/synset_words.txt
```

#### (Optional) Alarm sound
Any `.wav` file placed at `assets/alarm.wav` will play when inattention is detected.

---

### 4. Run the Phone / Face Detector

```bash
python phonedetection.py \
    --shape-predictor assets/shape_predictor_68_face_landmarks.dat \
    --prototxt assets/bvlc_googlenet.prototxt \
    --model assets/bvlc_googlenet.caffemodel \
    --labels assets/synset_words.txt \
    --alarm assets/alarm.wav
```

Requires a connected webcam.

---

### 5. Run the Audio Processor (optional)

Set your Google Cloud credentials before running:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account.json"
python audio.py
```

To get credentials: [Google Cloud Speech-to-Text Quickstart](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries)

---

## Tech Stack
`Python` `OpenCV` `dlib` `Caffe` `Apache Kafka` `Flask` `React.js` `Google Cloud Speech`

## Citation
If you use this work, please cite:
```
V. Karthikraj, V. Patil, S. Thanneermalai, et al.,
"Attention Span Detection for Online Lectures,"
IEEE ICACC 2021. https://ieeexplore.ieee.org/abstract/document/9299647
```