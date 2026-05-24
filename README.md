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
- Tracks face presence, eye contact, background noise, and dialogue ratio

## System Architecture
```
Video/Audio Stream
        │
   ┌────┴─────────────────┐
   │                       │
Phone Detection        Audio Analysis
(OpenCV + YOLO)       (background noise,
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

## Setup

### Prerequisites
```bash
pip install opencv-python kafka-python flask flask-cors numpy
```
Requires: Python 3.8+, Apache Kafka running locally or on a server.

### Running
```bash
# Start Kafka (ensure Zookeeper + Kafka broker are running)

# Run phone/face detection
python phonedetection.py

# Run audio processing
python audio.py
```

## Tech Stack
`Python` `OpenCV` `Apache Kafka` `Flask` `React.js` `NumPy`

## Citation
If you use this work, please cite:
```
V. Karthikraj, V. Patil, S. Thanneermalai, et al.,
"Attention Span Detection for Online Lectures,"
IEEE ICACC 2021. https://ieeexplore.ieee.org/abstract/document/9299647
```
