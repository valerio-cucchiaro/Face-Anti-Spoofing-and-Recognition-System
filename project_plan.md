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

* **face anti-spoofing** (fake face detection)
* **face recognition**

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

#### System Pipeline

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

Main technologies expected to be used:

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

## 3. Implementation Plan

### 3.1 Dataset

The dataset will consist of:

* facial images of real people
* images or samples representing spoofed faces (e.g. images or synthetic faces)

Possible sources include:

* public datasets
* controlled images captured with webcam

The dataset will be divided into:

* training set
* test set

---

### 3.2 Processing Pipeline

Example processing pipeline:

1. Load image
2. Convert to grayscale
3. Detect face using OpenCV
4. Normalize the detected face
5. Perform anti-spoofing analysis
6. If the face is genuine, extract features
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

### 3.3 Software Architecture

The implementation is organized as a modular system in order to keep the different processing stages independent.

The main modules of the system are:

* detection
* preprocessing
* antispoofing
* recognition
* evaluation

A dedicated pipeline layer coordinates the execution of these modules.

During development three pipeline versions are maintained:

* run_pipeline_full: complete system used for development and integration
* run_pipeline_cv: simplified pipeline used for the Computer Vision project
* run_pipeline_bs: biometric pipeline used for the Biometric Systems project

Each pipeline is executed through a minimal entry-point script:

* main.py → run_pipeline_full
* main_cv.py → run_pipeline_cv
* main_bs.py → run_pipeline_bs

This structure allows reuse of the same modules while enabling different experimental configurations for the two courses.

### 3.4 Evaluation Methodology

The system will be evaluated using common biometric performance metrics:

* FAR (False Acceptance Rate)
* FRR (False Rejection Rate)
* GAR
* EER
* ROC curve

In the context of anti-spoofing:

* FAR represents a **fake face incorrectly accepted as genuine**
* FRR represents a **real user incorrectly rejected**

Dataset organization will follow a standard experimental protocol:

* training set
* testing set
* probe vs gallery split for recognition

---

## 4. Connection with Biometric Systems Course

The project strongly relates to several topics studied in the Biometric Systems course.

### Biometric System Architecture

The system follows the typical biometric processing pipeline including:

* acquisition
* preprocessing
* feature extraction
* template creation
* matching
* decision

An additional **presentation attack detection stage** is introduced for security.

---

### Presentation Attacks and Spoofing

Modern biometric systems must defend against attacks such as:

* printed photos
* screen replay attacks
* synthetic or AI-generated faces

The anti-spoofing module directly addresses this problem.

---

### Performance Evaluation

The following biometric evaluation metrics will be used:

* FAR
* FRR
* GAR
* EER
* ROC curves

These metrics allow a quantitative analysis of the biometric system performance.

---

### Face Biometrics

Relevant topics include:

* face localization
* facial feature extraction
* template matching
* identity verification

---

## 5. Connection with Computer Vision Course

The project also involves several key Computer Vision concepts.

### Image Processing Pipeline

The project applies a typical computer vision workflow including:

* image acquisition
* object detection
* image preprocessing
* feature extraction
* classification

---

### Face Detection

The project will implement face localization techniques using OpenCV detectors.

This step corresponds to the object detection stage in computer vision.

---

### Image Classification

The anti-spoofing module represents an image classification problem:

```text
real face
fake face
```

Machine learning techniques will be used to perform this classification.

---

### Pattern Recognition

The recognition module will apply pattern recognition techniques to match facial features with stored templates.

Possible approaches include PCA-based recognition or texture-based methods.

---

## 6. Strategy to Deliver Two Different Projects

Although the implementation is shared, the two exams will emphasize different aspects.

### Biometric Systems Version

Focus on:

* biometric authentication architecture
* anti-spoofing protection
* template matching
* security against presentation attacks
* biometric evaluation metrics

Additional tasks include:

* detailed FAR/FRR analysis
* performance comparison between configurations
* experimental biometric evaluation.

---

### Computer Vision Version

Focus on:

* image processing pipeline
* face detection algorithms
* feature extraction techniques
* classification of real vs fake faces
* discussion of the related scientific paper

Additional tasks include:

* analysis of visual challenges such as illumination or pose
* discussion of computer vision methods used in the project.

---

## 7. Shared Core Code Structure

The project will be organized as a modular system.

```text
project/
│
├── main.py                # entry point per eseguire la pipeline
│
├── dataset/               # raccolta immagini e gestione dataset
│
├── detection/             # face detection
│
├── preprocessing/         # crop, resize, normalization
│
├── antispoofing/          # presentation attack detection
│
├── feature_extraction/    # embeddings / feature vectors
│
├── recognition/           # matching / identity decision
│
├── evaluation/            # FAR, FRR, ROC, metrics
│
├── demo/                  # demo scripts o UI
│
├── models/                # modelli ML (haar, reti, ecc.)
│
└── utils/                 # funzioni comuni (camera, IO, ecc.)
```

The same implementation will support both courses, while the final reports will emphasize different theoretical aspects.
