# Development Log

## 2026-03-08

**Project Initialization**:

- created GitHub repository
- defined project idea and architecture
- connected local project to GitHub repository
- created general `.gitignore`

---

## 2026-03-09

**Environment Setup**:

- simplified and reorganized project ```README```
- created Python virtual environment (```venv```)
- installed initial project dependencies
- generated ```requirements.txt```
- created development log (```devlog.md```)
- downloaded `haarcascade_frontalface_default.xml` face detection model to `models/`

**Face Detection Pipeline (main.py)**:
Implemented the first functional pipeline for real-time face detection using OpenCV.
The script performs the following steps:

1. Access the system webcam using OpenCV VideoCapture.
2. Convert each captured frame to grayscale for faster and more reliable face detection.
3. Use a Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect faces.
4. Draw bounding boxes around detected faces in the original frame.
5. Extract the detected face region (ROI) from the grayscale frame.
6. Normalize the face by resizing it to a fixed resolution (128x128).

This establishes the base pipeline required for the later stages of the project:
webcam → face detection → face crop → face normalization

**Code Modularization**:

- refactored the initial `main.py` script into separate modules to improve code structure
- separated face detection logic into `detection/face_detector.py`
- separated image preprocessing functions into `preprocessing/face_preprocessing.py`
- kept `main.py` responsible only for pipeline orchestration and webcam handling
- added `__init__.py` files to define Python packages

Project structure after refactoring:

```text
project/
│
├── main.py
├── models/
│   └── haarcascade_frontalface_default.xml
├── detection/
│   ├── __init__.py
│   └── face_detector.py
└── preprocessing/
    ├── __init__.py
    └── face_preprocessing.py
```

---

## 2026-03-10

**Face Detection Improvements**:

Refined the face detection stage and improved runtime behavior of the webcam pipeline.

- implemented detection handling when no faces are found
- added on-screen feedback message ("No face detected")
- modified detection logic to select only the largest detected face when multiple faces are present
- added distance feedback to the user based on detected face size ("Move closer to the camera")
- introduced configurable detection parameters (`scaleFactor`, `minNeighbors`, `minSize`)
- exposed detection parameters as constants in `main.py` to simplify tuning

These changes improve detection stability and provide clearer feedback during webcam interaction.

**Face Preprocessing Improvements**:

Extended the preprocessing pipeline with illumination normalization.

- implemented histogram equalization (`cv2.equalizeHist`) on the detected face region
- added new preprocessing function `face_histogram_equalization()` to `preprocessing.py`

Updated preprocessing pipeline:

webcam frame → grayscale → face detection → face crop → resize → histogram equalization

This normalization step improves robustness for future face recognition and anti-spoofing stages by reducing sensitivity to lighting variations.

**Stage 1 completed**:
Implemented and stabilized the real-time webcam face detection and preprocessing pipeline, including largest-face selection, distance feedback, crop, resize, and histogram equalization. The codebase was modularized into main orchestration, detection, and preprocessing modules. The project is now ready to move to Stage 2: dataset collection.

**Stage 2: Dataset Collection Implementation**:

- created dataset module `src/dataset/dataset_collector.py`
- created dataset directory structure `data/recognition_train/<person>`
- implemented automatic dataset directory creation if missing
- implemented dataset indexing to continue numbering existing images
- integrated face detection and preprocessing pipeline inside dataset collector
- implemented face validation using minimum face area threshold
- implemented keyboard-controlled dataset capture (`s` key)
- implemented sequential image saving (`001.png`, `002.png`, ...)
- implemented maximum sample limit per person
- collected first dataset samples for initial testing (30 images)

The dataset collector reuses the same detection and preprocessing pipeline used in the main application to ensure consistency between training data and runtime recognition.

**Dataset Loading Implementation**:

Implemented the dataset loading module required for training the face recognition model.

- created dataset loader module `src/dataset/dataset_loader.py`
- implemented automatic scanning of all person directories inside `data/recognition_train/`
- implemented dynamic assignment of numeric labels to each person
- implemented label mapping (`label → person_name`)
- implemented loading of dataset images in grayscale using OpenCV
- implemented construction of `faces` and `labels` arrays required for training
- added basic validation to skip unreadable images

The dataset loader converts the dataset folder structure into the format required by OpenCV face recognizers (list of face images and corresponding numeric labels).

---

## 2026-03-11 — Project Refactor: Modular Pipeline Architecture

**Pipeline Architecture Refactor**:

- created new module `src/pipeline/` to centralize system orchestration
- created `frame_processing.py` for frame-level validation logic
- created `run_pipeline_full.py` implementing the full runtime pipeline
- implemented function `get_valid_face()` to validate detected faces
- added face validation steps: grayscale conversion, face detection, largest face selection, minimum face area check
- implemented `preprocess_face()` function in `src/preprocessing/face_preprocessing.py`
- centralized preprocessing pipeline: crop → resize → histogram equalization
- refactored the runtime pipeline to use modular detection, validation, and preprocessing stages
- created clean entry points:

  - `src/main.py`
  - `src/main_cv.py`
  - `src/main_bs.py`
- separated orchestration logic from processing modules

Runtime pipeline after refactor:

webcam acquisition
↓
face detection
↓
face validation
↓
face preprocessing
↓
display

This refactor establishes a clean modular architecture that will support the future integration of face recognition and anti‑spoofing modules.

---

## 2026-03-12

**Face Recognition Training Implementation**:

- created recognition module `src/recognition/`
- implemented training script `src/recognition/train_model.py`
- integrated dataset loader with recognition training pipeline
- initialized OpenCV LBPH face recognizer
- trained recognition model using dataset images
- implemented automatic creation of `models/` directory if missing
- saved trained model to `models/lbph_model.xml`
- saved label mapping to `models/label_map.pkl`
- updated project dependencies to use `opencv-contrib-python` (required for LBPH recognizer)

**Offline Recognition Testing**:

- implemented offline evaluation script `src/recognition/test_model.py`
- added functionality to load trained LBPH model from disk
- implemented loading of label mapping file
- implemented prediction of identity using test dataset images
- implemented confidence score reporting for each prediction
- implemented automatic accuracy calculation on test dataset
- validated recognition performance using reserved test images

**Recognition Evaluation Preparation**:

- collected confidence values for genuine matches
- prepared infrastructure for computing genuine and impostor score distributions
- prepared threshold-based recognition logic for handling unknown identities

Recognition decision rule prepared for runtime pipeline:

if confidence < threshold
→ recognized identity
else
→ Unknown

This stage completes the offline face recognition workflow and prepares the system for integration of real‑time recognition into the webcam pipeline.

**Real-Time Face Recognition Integration**:

- created recognition runtime module `src/recognition/face_recognizer.py`
- implemented function `load_recognizer()` to load the trained LBPH model and label map
- implemented function `predict_identity()` to perform identity prediction
- implemented threshold-based classification to handle unknown identities
- integrated recognition stage into the full webcam pipeline (`run_pipeline_full.py`)
- implemented real-time identity prediction on detected faces
- added visualization of predicted identity on the webcam stream
- added visualization of LBPH confidence score
- implemented runtime classification rule:

confidence < threshold → recognized identity  
confidence ≥ threshold → Unknown

- validated recognition behavior using partial occlusions and distance variations
- confirmed that high confidence values correctly trigger "Unknown" classification
- added module and function docstring

---

## 2026-03-16

**Environment Fix and Dependency Update**:

- detected incompatibility between Python 3.14 environment and OpenCV bindings (`cv2.imread` missing)
- installed Python **3.12** for stable compatibility with OpenCV
- recreated project virtual environment using Python 3.12
- installed project dependencies compatible with computer vision stack
- verified correct OpenCV installation and availability of `cv2.imread` and `cv2.face`
- updated project environment for future anti-spoofing development

**Anti-Spoofing Dataset Preparation**:

- selected **CASIA-FASD** dataset for presentation attack detection training
- downloaded dataset (train/test split with `live` and `spoof` samples)
- extracted dataset locally and integrated it into project structure
- created dataset directory `data/casia_pad/`
- verified image loading using OpenCV (`cv2.imread`)
- verified dataset image format and resolution (`112x112`)

The dataset will be used to train and evaluate the texture-based anti-spoofing module before integration into the runtime pipeline.

**Anti-Spoofing Feature Extraction**:

- created module `src/antispoofing/texture_features.py`
- implemented LBP-based texture feature extraction
- generated normalized LBP histogram (10-dimensional feature vector)
- created module `src/antispoofing/quality_features.py`
- implemented basic image quality descriptors (mean, std, blur, edge density, entropy)
- generated quality feature vector (5 features)
- created module `src/antispoofing/feature_extractor.py`
- implemented `extract_pad_features()` to combine LBP and quality descriptors
- produced final PAD feature vector of **15 features**

**Feature Extraction Testing**:

- created testing script `src/antispoofing/test_features.py`
- implemented functions to test texture, quality and combined features
- loaded random images from CASIA-FASD dataset for validation
- verified feature vector shapes and histogram normalization

Output verification:

- LBP feature vector: `(10,)`
- Quality feature vector: `(5,)`
- Combined PAD feature vector: `(15,)`

**Anti-Spoofing Training Pipeline Implementation**:

Extended the anti-spoofing module by implementing the PAD training pipeline using the CASIA-FASD dataset.

- created PAD dataset loader `src/antispoofing/pad_dataset_loader.py`
- implemented loading of live and spoof samples from `data/casia_pad/train/`
- implemented feature extraction for each dataset image using the PAD feature extractor
- constructed training feature matrix `X_train` and label vector `y_train`
- created classifier module `src/antispoofing/svm_classifier.py`
- implemented feature normalization using `StandardScaler`
- implemented SVM classifier (`kernel=rbf`, `class_weight=balanced`, `probability=True`)
- created training pipeline `src/antispoofing/train_pad.py`
- integrated dataset loading and classifier training
- implemented dataset statistics reporting during training
- implemented automatic creation of `models/` directory
- saved trained PAD artifacts:

  - `models/pad_svm.joblib`
  - `models/pad_scaler.joblib`

The system can now train and store a complete PAD classifier for later runtime inference.

**Anti-Spoofing Feature Improvements and Evaluation**:

Refined the PAD feature extraction pipeline to improve spoof detection capability and feature robustness.

- expanded LBP texture descriptor to multi-region histogram (54 features)
- simplified quality feature set to stable statistical descriptors
- implemented Local Feature Transformation (LFT) descriptor
- created module `src/antispoofing/lft_features.py`
- implemented computation of LFT map using local mean and variance normalization
- implemented statistical descriptors from LFT map (mean, std, skewness, kurtosis)
- integrated LFT features into PAD feature extractor
- updated PAD feature vector size from **15 → 64 features**
- updated testing script `src/antispoofing/test_antispoofing_features.py`
- verified feature extraction pipeline correctness and vector dimensions

Final PAD feature vector composition:

LBP texture features → 54  
Quality features → 6  
LFT features → 4  

Total PAD feature vector size → **64 features**

**PAD Model Retraining and Evaluation**:

- retrained SVM PAD classifier using updated feature set
- trained on **51,793 samples** from CASIA-FASD dataset
- evaluated model on **54,507 test samples**

Evaluation results:

Accuracy → **0.8849**  
Precision → **0.6434**  
Recall → **0.7533**  
F1 Score → **0.6940**

These results show improved recall and overall F1 score compared to the previous PAD configuration, confirming that the introduction of LFT features improves spoof detection performance.

---

## 2026-03-17

**Anti-Spoofing Module Refactor (Training vs Runtime)**:

- reorganized the anti-spoofing package to separate offline training/evaluation code from runtime inference code
- created `src/antispoofing/training/` to contain PAD training, evaluation, dataset loading, runtime sample collection, and offline runtime testing
- kept runtime PAD components at the root of `src/antispoofing/`
- introduced `src/antispoofing/pad_inference_utils.py` to isolate runtime feature extraction from an already cropped face
- updated PAD imports and filesystem paths after the refactor
- renamed training helper `extract_face_features_for_img()` to `extract_pad_features_from_img()` for consistency with the PAD naming

Updated anti-spoofing structure:

```text
src/antispoofing/
├── spoof_detector.py
├── pad_inference_utils.py
├── feature_extractor.py
├── texture_features.py
├── quality_features.py
├── lft_features.py
└── training/
    ├── train_pad.py
    ├── evaluate_pad.py
    ├── pad_dataset_loader.py
    ├── svm_classifier.py
    ├── collect_pad_runtime.py
    ├── test_runtime.py
    └── test_antispoofing_features.py
```

This refactor cleanly separates the offline PAD workflow from the runtime biometric pipeline while preserving shared feature extraction code.

**PAD Runtime Inference Implementation**:

- completed runtime anti-spoofing detector in `src/antispoofing/spoof_detector.py`
- implemented runtime loading of:

  - `models/pad_svm.joblib`
  - `models/pad_scaler.joblib`

- implemented `AntiSpoofingDetector.predict()` returning:

  - PAD status (`ok` / `invalid_input`)
  - binary prediction
  - human-readable label (`live` / `spoof` / `unknown`)
  - confidence score

- implemented runtime feature extraction from face crops in `pad_inference_utils.py`
- aligned runtime PAD preprocessing with training by using grayscale face crops and `preprocess_face_for_pad()`

Runtime PAD flow:

webcam frame
→ face detection
→ grayscale face crop
→ PAD preprocessing
→ PAD feature extraction
→ scaler
→ SVM prediction

**Full Pipeline Integration of PAD**:

- integrated anti-spoofing inference into `src/pipeline/run_pipeline_full.py`
- added loading of the PAD detector together with the recognition model
- inserted PAD decision stage before face recognition
- blocked recognition whenever the final PAD decision is not `live`
- added real-time visualization of PAD result on the webcam stream
- used grayscale crop for PAD in runtime to better match the training pipeline

**Temporal Smoothing and Runtime Decision Logic**:

- introduced temporal smoothing for PAD decisions using a fixed-length `deque`
- added configurable smoothing parameters:

  - `PAD_WINDOW_SIZE`
  - `PAD_MIN_LIVE`
  - `PAD_MIN_SPOOF`

- implemented dual PAD output in the runtime pipeline:

  - current-frame PAD result
  - final smoothed PAD decision

- introduced explicit `unknown` state when the temporal evidence is insufficient or the current PAD input is invalid
- cleared PAD history when no valid face is detected

Decision rule in the runtime pipeline:

- enough recent `live` predictions → final PAD = `live`
- enough recent `spoof` predictions → final PAD = `spoof`
- otherwise → final PAD = `unknown`

**Runtime PAD Dataset Collection and Offline Runtime Evaluation**:

- created `src/dataset/collect_pad_runtime.py`
- implemented webcam-based collection of real runtime PAD samples
- saved grayscale face crops into:

  - `data/pad_runtime_test/live`
  - `data/pad_runtime_test/spoof`

- added keyboard-controlled collection:

  - `l` for live samples
  - `s` for spoof samples
  - `q` to quit

- created `src/antispoofing/training/test_runtime.py`
- implemented offline evaluation of the trained PAD model on collected webcam samples
- reused the same preprocessing and feature extraction pipeline used during PAD training
- added per-image diagnostic output including prediction, class probabilities, feature mean, and feature standard deviation

Runtime test summary on collected webcam samples:

- total images: `60`
- valid processed images: `54`
- correct predictions: `43`
- accuracy: `0.7963`
- live samples: `17 / 28` correctly classified
- spoof samples: `26 / 26` correctly classified

These results showed that the trained PAD model generalizes well to spoof webcam attacks, but remains significantly more conservative on real live webcam samples, often preferring the `spoof` class under uncertainty.

---

## 2026-03-19

**Replay-Attack Dataset Integration for PAD**:

- integrated Replay-Attack as a second PAD dataset alongside CASIA-FASD
- created raw dataset folder `data/replay_attack_raw/`
- created extracted PAD image dataset `data/replay_attack_pad/`
- organized Replay-Attack PAD data into:

  - `train/live`
  - `train/spoof`
  - `test/live`
  - `test/spoof`

- created frame extraction utility `src/dataset/extract_replayattack_frames.py`
- implemented automatic split mapping:

  - `devel` + `train` -> PAD `train`
  - `test` -> PAD `test`

- implemented automatic label mapping:

  - `real` -> `live`
  - `attack` -> `spoof`

- implemented unique frame naming to avoid filename collisions across videos
- sampled frames from Replay-Attack videos every fixed interval for PAD training

This extends the anti-spoofing training data with more realistic replay/display attacks and a broader domain than CASIA alone.

**PAD Loader and Experiment Pipeline Update**:

- extended `src/antispoofing/training/pad_dataset_loader.py` to support:

  - CASIA-only loading
  - Replay-only loading
  - combined CASIA + Replay evaluation

- added `load_pad_dataset_replay()` for Replay-only PAD training
- added Replay-specific evaluation split loading without balancing on the test set
- updated `src/antispoofing/training/train_pad.py` to allow Replay-only training by selecting `load_pad_dataset_replay()`
- updated `src/antispoofing/training/evaluate_pad.py` to support selectable evaluation modes:

  - `casia`
  - `replay`
  - `combined`

- created `src/antispoofing/training/run_experiment.py` to automate train + evaluate PAD experiments

This made it possible to compare:

- Replay-only training
- CASIA + Replay combined training
- cross-dataset generalization between Replay and CASIA

**Multi-Scale PAD Feature Extraction**:

- upgraded `src/antispoofing/feature_extractor.py` from single-scale to multi-scale PAD extraction
- separated the original descriptor into `extract_pad_features_single_scale()`
- kept the original scale features on the preprocessed face image
- added a second scale by applying:

  - Gaussian blur
  - downsampling to `64x64`

- concatenated original-scale and downsampled-scale descriptors into a single multi-scale PAD vector

This multi-scale extension was introduced to better capture spoof artefacts appearing at different spatial frequencies, such as blur, display noise, and texture inconsistencies.

**PAD Experiments with Replay-Attack and Multi-Scale Features**:

Replay-only training before multi-scale:

- evaluated on Replay-Attack test set
- Accuracy: `0.9547`
- Precision: `0.9323`
- Recall: `0.8764`
- F1 Score: `0.9035`

- evaluated on CASIA test set for cross-dataset generalization
- Accuracy: `0.6036`
- Precision: `0.0319`
- Recall: `0.0439`
- F1 Score: `0.0370`

Combined CASIA + Replay training before multi-scale:

- evaluated on combined test set
- Accuracy: `0.8930`
- Precision: `0.6851`
- Recall: `0.7862`
- F1 Score: `0.7321`

Combined CASIA + Replay training after multi-scale:

- evaluated on combined test set
- Accuracy: `0.9204`
- Precision: `0.7723`
- Recall: `0.8112`
- F1 Score: `0.7913`

Replay-only training after multi-scale:

- evaluated on CASIA test set
- Accuracy: `0.5699`
- Precision: `0.0660`
- Recall: `0.1128`
- F1 Score: `0.0833`

These experiments confirmed two main points:

- Replay-Attack training alone performs very well on Replay-style attacks but generalizes poorly to CASIA
- combining CASIA and Replay and then introducing multi-scale PAD features significantly improves the overall PAD results on the joint evaluation setup

**PAD Naming Refactor and File Reorganization**:

Refactored the anti-spoofing PAD file naming to improve clarity and consistency as the project structure grew.

Renamed PAD training and experiment files:

- `src/antispoofing/training/train_pad.py` -> `src/antispoofing/training/pad_model_trainer.py`
- `src/antispoofing/training/evaluate_pad.py` -> `src/antispoofing/training/pad_model_evaluator.py`
- `src/antispoofing/training/run_experiment.py` -> `src/antispoofing/training/pad_experiment_runner.py`
- `src/antispoofing/training/svm_classifier.py` -> `src/antispoofing/training/pad_svm_trainer.py`
- `src/antispoofing/training/pad_dataset_loader.py` -> `src/antispoofing/training/pad_training_dataset_loader.py`
- `src/antispoofing/training/test_runtime.py` -> `src/antispoofing/training/pad_runtime_evaluator.py`
- `src/antispoofing/training/test_antispoofing_features.py` -> `src/antispoofing/training/pad_feature_extractor_tester.py`

Renamed PAD feature extraction and runtime files:

- `src/antispoofing/feature_extractor.py` -> `src/antispoofing/pad_feature_extractor.py`
- `src/antispoofing/texture_features.py` -> `src/antispoofing/pad_texture_features.py`
- `src/antispoofing/quality_features.py` -> `src/antispoofing/pad_quality_features.py`
- `src/antispoofing/lft_features.py` -> `src/antispoofing/pad_lft_features.py`
- `src/antispoofing/pad_inference_utils.py` -> `src/antispoofing/pad_inference_preprocessing.py`
- `src/antispoofing/spoof_detector.py` -> `src/antispoofing/pad_runtime_detector.py`

Renamed PAD dataset utility files:

- `src/dataset/collect_pad_runtime.py` -> `src/dataset/pad_runtime_dataset_collector.py`
- `src/dataset/extract_replayattack_frames.py` -> `src/dataset/replayattack_frame_extractor.py`

This refactor established a more explicit naming convention across the PAD module:

- `pad_*` prefix for anti-spoofing specific components
- `*_trainer`, `*_evaluator`, `*_loader`, `*_collector`, `*_extractor`, `*_runner` suffixes to make each file role immediately recognizable

Final cleanup after the rename:

- updated all affected imports and internal references across the PAD training, runtime, dataset, and pipeline modules
- updated runtime pipeline references after renaming the PAD detector module
- aligned module docstrings with the new filenames
- cleaned function docstrings to remain consistent with the project style
- added or refined comments in the renamed files where the behavior was less immediate
- verified that filesystem paths still resolved correctly after file moves
- verified syntax correctness of the refactored modules

---

## IN FASE DI IMPLEMENTAZIONE

1. Improve recognition robustness with multiple identities in the dataset

2. Collect dataset samples for additional users

3. Evaluate recognition performance and tune UNKNOWN_THRESHOLD

4. Implement runtime PAD detector module

5. Integrate anti-spoofing into the webcam pipeline

6. Combine recognition and anti-spoofing decisions

7. Evaluate full system performance (recognition + PAD)
