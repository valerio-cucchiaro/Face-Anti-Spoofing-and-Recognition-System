# Face Anti-Spoofing and Recognition System

This project implements a biometric face authentication system that combines **fake face detection (anti-spoofing)** and **face recognition** using computer vision techniques.

The system first determines whether a detected face is **genuine or spoofed** (for example AI-generated or presentation attacks).
Only if the face is considered genuine, the system proceeds with **identity recognition**.

The goal is to demonstrate how modern biometric systems integrate **security mechanisms against spoofing attacks** before performing identity verification.

---

## System Overview

Typical system workflow:

```text
image / webcam input
        ↓
face detection
        ↓
preprocessing
        ↓
fake face detection (anti-spoofing)
        ↓
(real face?)
   ↓            ↓
 reject     face recognition
                  ↓
            identity decision
```

The system therefore performs two main tasks:

* **Fake Face Detection** (anti-spoofing)
* **Face Recognition**

---

## Objectives

The project aims to:

* detect human faces in images or video streams
* distinguish **real faces from spoofed or synthetic faces**
* prevent presentation attacks in biometric systems
* extract facial features
* perform identity recognition when the face is genuine
* evaluate system performance using biometric metrics

---

## Technologies

Planned technologies:

Programming language:

* Python

Libraries:

* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

Optional (depending on experiments):

* TensorFlow
* PyTorch

---

## Project Structure

Planned modular structure:

```text
project/
│
├── dataset/
├── detection/
├── preprocessing/
├── antispoofing/
├── feature_extraction/
├── recognition/
├── evaluation/
└── demo/
```

Each module will correspond to a specific stage of the biometric processing pipeline.

---

## Relation to Biometric Systems

This project implements a simplified **biometric authentication system** including:

* acquisition
* preprocessing
* feature extraction
* template matching
* identity decision

Additionally, the system includes a **presentation attack detection (PAD)** module that determines whether the face presented to the system is genuine or spoofed.

Performance will be evaluated using biometric metrics such as:

* FAR (False Acceptance Rate)
* FRR (False Rejection Rate)
* EER
* ROC curves

---

## Relation to Computer Vision

From the computer vision perspective, the project involves:

* face detection
* image preprocessing
* feature extraction
* image classification (real vs fake face)
* pattern recognition for identity matching

These steps follow a standard computer vision pipeline applied to facial image analysis.

---

## Project Status

Initial setup and system design phase.
