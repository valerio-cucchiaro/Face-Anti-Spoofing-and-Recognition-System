# Face Anti-Spoofing and Recognition System

## 1. Exam Requirements

### 1.1 Biometric Systems Requirements

The Biometric Systems exam requires the development of a biometric module including:

* System architecture description
* Implementation using MATLAB or OpenCV
* Experimental evaluation of the system
* Performance analysis
* Written report and presentation
* Working demo of the system

The project must demonstrate:

* a biometric processing pipeline
* secure biometric authentication concepts
* performance evaluation and experimental analysis

In this project the biometric system will include an additional **presentation attack detection stage** in order to determine whether the presented face is genuine or spoofed before performing identity recognition.

---

### 1.2 Computer Vision Requirements

The Computer Vision exam consists of:

1. Paper presentation (10 minutes)
2. Project related to the chosen paper (10 minutes)

The project:

* requires some coding
* can reuse existing code
* must be coherent with the problem discussed in the selected paper
* does not require implementing a system completely from scratch

The implementation will focus on image analysis and classification techniques applied to facial images.

---

## 2. Project Idea

### 2.1 General Description

The proposed project is a **Face Anti-Spoofing and Recognition System** based on Computer Vision and Biometric System techniques.

The system will detect a human face from an image or webcam stream and determine whether the presented face is:

```text
REAL (genuine user)
FAKE (spoofed or AI-generated face)
```

If the system determines that the face is genuine, it will then attempt to recognize the identity of the subject using a face recognition module.

The project therefore integrates two main functionalities:

* face anti-spoofing (fake face detection)
* face recognition

This structure reflects real biometric authentication systems, where security mechanisms are used to prevent spoofing attacks before identity verification.

---

### 2.2 Objectives

The main objectives of the system are:

* detect human faces in images or video streams
* determine whether the face is genuine or spoofed
* prevent presentation attacks against the biometric system
* extract facial features for identity recognition
* compare the extracted features with stored biometric templates
* identify or verify the identity of the subject
* evaluate system performance using biometric metrics

---

### 2.3 System Architecture

The system will follow an extended biometric recognition pipeline.

Processing pipeline:

1. Image acquisition
2. Face detection
3. Face preprocessing
4. Anti-spoofing analysis
5. Feature extraction
6. Template creation
7. Matching with database
8. Identity decision
9. Performance evaluation

Logical structure of the system:

```text
camera / image
      ↓
face detection
      ↓
preprocessing
      ↓
anti-spoofing module
      ↓
(real face?)
   ↓        ↓
 no         yes
reject   face recognition
             ↓
        identity decision
```

---

### 2.4 Technologies Used

Programming language:

* Python

Libraries:

* OpenCV
* NumPy
* scikit-image
* scikit-learn
* MediaPipe
* matplotlib

Additional utilities:

* joblib

---

## 3. Implementation Plan

### 3.1 Dataset

The dataset will consist of:

* facial images of real people
* images or videos representing spoofed faces

Possible sources include:

* CASIA Face Anti-Spoofing Dataset
* Replay-Attack Dataset

These datasets include common presentation attacks such as:

* printed photo attacks
* replay attacks using screens

In addition to public datasets, a **small custom dataset** may be collected using a webcam including:

* genuine faces
* printed photographs
* replay attacks on smartphone screens
* AI-generated faces displayed on screens

The dataset will be divided into:

* training set
* validation set
* testing set

---

### 3.2 Processing Pipeline

Example processing pipeline:

1. Load image or capture webcam frame
2. Convert to grayscale
3. Detect face using OpenCV
4. Normalize the detected face (crop, resize, illumination normalization)
5. Perform anti-spoofing analysis
6. If the face is genuine, extract recognition features
7. Train recognition classifier
8. Perform identity recognition

Possible algorithms for recognition include:

* Eigenfaces (PCA)
* LBPH
* classical machine learning classifiers such as SVM or KNN

The anti-spoofing module will perform a classification between:

```text
real face
fake face
```

---

### 3.3 Anti-Spoofing Methodology

The anti-spoofing module is inspired by the selected research paper on **face presentation attack detection based on multi-scale image quality assessment**.

The central idea is that spoofed faces often contain **perceptual quality artifacts** introduced during the printing or screen recapturing process.

Typical artifacts include:

* luminance distortions
* blur and defocus
* contrast alterations
* texture inconsistencies
* recapturing noise

These artifacts can be detected using **hand-crafted image features combined with a classifier**.

The anti-spoofing pipeline will therefore follow this structure:

```text
Face image
      ↓
Feature extraction
      ↓
Feature vector
      ↓
Machine learning classifier
      ↓
real / spoof decision
```

---

### 3.4 Anti-Spoofing Feature Extraction

Three groups of features will be extracted from each detected face.

#### Texture Features

Texture descriptors capture local spatial variations in facial images.

The main descriptor used will be:

* Local Binary Pattern (LBP)

LBP histograms provide robust texture representations and have been widely used in face anti-spoofing research.

These features help detect:

* printing artifacts
* screen pixel patterns
* texture inconsistencies

---

#### Image Quality Features

Image quality metrics capture distortions introduced during spoof attacks.

Examples include:

* variance of Laplacian (blur detection)
* contrast statistics
* luminance distribution
* edge density
* high-frequency energy

These features model degradation effects caused by the **recapturing process**.

---

#### Temporal Features

Temporal cues help detect replay attacks using video sequences.

Examples include:

* blink detection
* head motion consistency
* short temporal motion analysis

Temporal analysis helps distinguish real human faces from static images or replayed videos.

---

### 3.5 Feature Vector

All extracted features will be combined into a single feature vector.

Example structure:

```text
[
LBP histogram bins,
blur score,
contrast score,
luminance statistics,
edge density,
noise level,
blink indicator,
temporal motion score
]
```

This feature vector represents the input for the classification stage.

---

### 3.6 Classification Model

The anti-spoofing classification will be performed using:

```text
Support Vector Machine (SVM)
```

Reasons for this choice:

* effective for medium-size feature vectors
* widely used in traditional PAD systems
* simpler than training deep neural networks from scratch

Suggested configuration:

* RBF kernel
* grid search for hyperparameter tuning

The classifier will output:

```text
real
spoof
```

---

### 3.7 Software Architecture

The implementation will follow a modular architecture.

Main modules:

* detection
* preprocessing
* antispoofing
* recognition
* evaluation

A pipeline module coordinates the execution of the system.

Three pipeline configurations will be maintained:

* run_pipeline_full → complete system
* run_pipeline_cv → Computer Vision version
* run_pipeline_bs → Biometric Systems version

Entry points:

* main.py
* main_cv.py
* main_bs.py

---

### 3.8 Evaluation Methodology

The system will be evaluated using biometric performance metrics:

* FAR (False Acceptance Rate)
* FRR (False Rejection Rate)
* GAR
* EER
* ROC curves

For anti-spoofing evaluation:

* FAR represents a fake face incorrectly accepted as genuine
* FRR represents a real user incorrectly rejected

Dataset organization will follow a standard protocol:

* training set
* testing set
* gallery vs probe split for recognition

---

## 4. Connection with Biometric Systems Course

Relevant topics include:

* biometric system architecture
* face recognition
* presentation attack detection
* biometric performance evaluation

The project demonstrates the full biometric pipeline with the addition of a **presentation attack detection stage**.

---

## 5. Connection with Computer Vision Course

Relevant Computer Vision concepts include:

* image acquisition
* object detection
* image preprocessing
* feature extraction
* image classification

The anti-spoofing module represents a **binary image classification problem**.

```text
real face
fake face
```

---

## 6. Strategy to Deliver Two Different Projects

Although the implementation is shared, the two exams will emphasize different aspects.

### Biometric Systems Version

Focus on:

* biometric authentication architecture
* anti-spoofing protection
* template matching
* presentation attack detection
* biometric performance metrics

---

### Computer Vision Version

Focus on:

* image processing pipeline
* face detection
* feature extraction
* real vs spoof classification
* discussion of the selected research paper

---

## 7. Shared Core Code Structure

The project will be organized as a modular system.

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

The same implementation supports both courses while reports highlight different theoretical aspects.
