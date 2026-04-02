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

**PAD Runtime Pipeline Refactor**:

Refactored the runtime PAD logic inside the full webcam pipeline to better separate orchestration, frame quality checks, and temporal state management.

- created `src/antispoofing/pad_frame_quality.py`
- implemented `evaluate_pad_crop_quality()` to isolate runtime PAD crop validation and blur estimation
- moved the Laplacian-based quality check out of `src/pipeline/run_pipeline_full.py`
- created `src/antispoofing/pad_runtime_state.py`
- implemented `PadRuntimeState.update()` to manage:

  - PAD temporal history
  - missing-face handling
  - averaged live probability
  - hysteresis between `unknown`, `live`, and `spoof`

- updated `src/pipeline/run_pipeline_full.py` to use the new PAD runtime helpers
- reduced the amount of PAD state logic embedded directly inside the pipeline loop
- kept `run_pipeline_full.py` focused on:

  - webcam acquisition
  - face validation
  - PAD inference orchestration
  - recognition gating
  - runtime visualization

- preserved the runtime behavior while making the PAD pipeline easier to read and maintain
- verified imports, function calls, and syntax correctness after the refactor

**Runtime Face Detection Update (MediaPipe Integration)**:

Updated the runtime and dataset collection face detection flow to support MediaPipe face detection in addition to the existing Haar Cascade backend.

- added `mediapipe==0.10.21` to `requirements.txt`
- extended `src/detection/face_detector.py` with:

  - `load_mediapipe_detector()`
  - `detect_faces_mediapipe()`
  - backend selection inside `detect_faces(...)`

- updated `src/preprocessing/frame_preprocessing.py` so `get_valid_face()` can work with different detection backends through a `method` parameter
- switched `src/pipeline/run_pipeline_full.py` to use MediaPipe face detection for the runtime webcam pipeline
- switched `src/dataset/dataset_collector.py` to use MediaPipe during recognition dataset collection
- switched `src/dataset/pad_runtime_dataset_collector.py` to use MediaPipe during PAD runtime dataset collection
- aligned PAD runtime crop extraction with the current runtime pipeline by adding a margin around the detected face before saving PAD test crops
- refined comments and docstrings in the updated face detection module

This update makes the runtime face localization more flexible and prepares the codebase to compare Haar Cascade and MediaPipe detection more easily.

---

## 2026/03/23

**MediaPipe-Only Face Detection Cleanup and Recognition Dataset Naming Update**:

Completed the cleanup after the MediaPipe migration and aligned the latest PAD training/runtime dataset changes.

- simplified `src/detection/face_detector.py` to keep only the MediaPipe detector loader and detection flow
- removed the remaining Haar Cascade compatibility logic from detection and preprocessing
- simplified `src/preprocessing/frame_preprocessing.py` so `get_valid_face()` now uses the shared MediaPipe flow directly
- updated all `get_valid_face()` call sites to use the simplified signature
- refined module and function docstrings in the updated detection, preprocessing, dataset, and pipeline files to match the MediaPipe-only behavior

- created `src/antispoofing/pad_crop_utils.py` to centralize the shared padded face crop logic used across runtime PAD inference, PAD runtime dataset collection, and offline PAD dataset loading

- replaced `src/antispoofing/training/pad_training_dataset_loader.py` with the clearer `src/antispoofing/training/pad_dataset_loader.py`
- updated PAD training and evaluation imports to use the new loader module
- integrated `data/pad_runtime_dataset` into the PAD training pipeline
- added runtime-specific PAD training support through:

  - `runtime`
  - `combined_runtime`

- introduced runtime oversampling in combined PAD training through `RUNTIME_REPEAT = 5` so the custom runtime dataset has more influence during training
- updated `src/antispoofing/training/pad_experiment_runner.py` to use the new combined runtime training setup
- updated `src/antispoofing/training/pad_runtime_evaluator.py` to evaluate already-cropped PAD runtime samples from `data/pad_runtime_dataset/test`
- updated `src/dataset/pad_runtime_dataset_collector.py` to save runtime PAD samples into `data/pad_runtime_dataset/train`
- aligned `.gitignore` with the new `data/pad_runtime_dataset` structure
- extended `src/antispoofing/pad_frame_quality.py` so PAD quality checks now include crop size, aspect ratio, and blur
- updated `src/pipeline/run_pipeline_full.py` to use the shared PAD crop helper and to skip PAD history updates when the face bounding box is unstable

Renamed the recognition dataset utility files for clarity:

- `src/dataset/dataset_collector.py` -> `src/dataset/recognition_dataset_collector.py`
- `src/dataset/dataset_loader.py` -> `src/dataset/recognition_dataset_loader.py`

Final consistency updates:

- updated `src/recognition/train_model.py` to import `load_dataset()` from `src/dataset/recognition_dataset_loader.py`
- aligned the face-size threshold to `8000` in the recognition dataset collector so it matches:

  - `src/pipeline/run_pipeline_full.py`
  - `src/dataset/pad_runtime_dataset_collector.py`
  - `src/antispoofing/training/pad_dataset_loader.py`

- normalized the remaining `cv2.putText(...)` calls so they are written on a single line
- verified imports, function calls, docstrings, and syntax correctness after the cleanup

---

## 2026/03/24

**PAD Pipeline Freeze and Runtime Dataset Alignment**:

Completed the PAD pipeline freeze to align collector, training, and runtime around a shared `final_pad_crop` contract.

- updated `src/antispoofing/training/pad_dataset_loader.py` to distinguish explicitly between `raw_frame` and `final_pad_crop`
- fixed the runtime PAD training dataset path so `data/pad_runtime_dataset` samples are no longer reprocessed as raw frames
- refactored the PAD loader so both dataset branches now converge into the same canonical post-crop path in `src/antispoofing/pad_inference_preprocessing.py`
- updated `src/antispoofing/pad_frame_quality.py` to centralize shared PAD crop-quality checks and to separate `evaluate_bbox_stability()` as runtime-only logic
- updated `src/dataset/pad_runtime_dataset_collector.py` and `src/pipeline/run_pipeline_full.py` to use the same shared bbox stability policy

Freeze verification after the refactor:

- `crop path closed`
- `preprocess path closed`
- `feature path closed`

- verified imports, function calls, docstrings, and syntax correctness after the freeze cleanup

---

## 2026/03/27

**PAD Training Sampling and Runtime Evaluation Debugging**:

- added deterministic PAD training sampling through `TRAIN_SAMPLING_SEED` and `CASIA_SAMPLE_RATIO`
- added training composition reporting for CASIA, Replay-Attack, and runtime samples
- aligned `src/antispoofing/training/pad_experiment_runner.py` and `src/antispoofing/training/pad_model_evaluator.py` to support runtime-focused PAD evaluation
- updated `src/antispoofing/training/pad_model_trainer.py` to print the active PAD debug feature mode
- set `RUNTIME_REPEAT = 10`

These changes made PAD experiments more reproducible and easier to compare on the runtime split.

**Runtime PAD Dataset Split Control**:

- updated `src/dataset/pad_runtime_dataset_collector.py` to support both `train` and `test` runtime PAD splits
- added configurable split selection and separate live/spoof collection limits
- showed the active split inside the collector UI

This made runtime PAD data collection easier to manage during debugging.

**PAD Feature Ablation Support**:

- added `PAD_DEBUG_FEATURE_MODE` in `src/antispoofing/pad_feature_extractor.py`
- supported debug modes:

  - `full`
  - `lbp_only`
  - `lbp_lft`
  - `no_quality`
  - `no_down_quality`

- kept the ablation logic isolated inside the feature extractor

This supported controlled runtime-focused PAD feature ablation tests.

**TGSD Prototype Integration**:

- created `src/antispoofing/pad_tgsd_features.py`
- implemented a first TGSD-style gradient discrepancy feature
- integrated TGSD into `src/antispoofing/pad_feature_extractor.py`
- updated feature block indexing after adding the new feature

PAD feature dimensions after TGSD:

- single scale: `65`
- full two-scale vector: `130`

**PAD Runtime Ablation Results**:

Initial feature ablations showed:

- `full` remained the best baseline
- `lbp_only` performed worst on runtime live samples
- `lbp_lft` improved over `lbp_only` but stayed below `full`
- `no_down_quality` stayed close to `full` but did not improve it

First TGSD result on runtime evaluation:

- `full` without TGSD:

  - Accuracy: `0.776`
  - Recall: `0.35`
  - F1 Score: `0.50`
  - live correct: `14/40`
  - spoof correct: `83/85`

- `full` with TGSD:

  - Accuracy: `0.792`
  - Recall: `0.40`
  - F1 Score: `0.5517`
  - live correct: `16/40`
  - spoof correct: `83/85`

The first TGSD prototype improved runtime live recall without reducing spoof accuracy.

---

## 2026/03/30

**Recognition Dataset Structure Refactor**:

- replaced the old custom recognition folders `data/recognition_train/` and `data/recognition_test/` with the new biometric structure `data/recognition/custom/gallery/` and `data/recognition/custom/probe/`
- created the full recognition dataset tree under `data/recognition/` for `custom`, `orl`, and `lfw`, including `raw`, `gallery`, and `probe` subfolders for the public datasets
- migrated the existing custom identity `valerio` into the new `gallery` / `probe` layout
- updated `src/dataset/recognition_dataset_collector.py` so it now writes directly to the new custom recognition structure and uses `gallery` / `probe` modes instead of the older `train` / `test` naming
- kept separate collection limits for the two custom splits: `30` gallery images and `10` probe images

This refactor aligns the recognition module with the biometric `gallery / probe` terminology used in the course and removes the older duplicate folder convention.

**Recognition Public Dataset Preparation Utilities**:

- created `src/dataset/orl_dataset_splitter.py`
- implemented fixed ORL split generation with images `1-5` -> gallery and images `6-10` -> probe
- preserved the original ORL dataset under `data/recognition/orl/raw`
- made the ORL split generation idempotent by clearing old outputs before recreating them
- added validation for malformed ORL identity folders and incomplete subject folders
- created `src/dataset/lfw_dataset_splitter.py`
- implemented LFW identity filtering based on a minimum number of valid images
- implemented dynamic per-identity gallery / probe splitting for LFW
- preserved the original LFW dataset under `data/recognition/lfw/raw`
- added summary reporting for processed and skipped LFW identities

This established a clean and reproducible preparation path for both public recognition datasets while keeping the raw sources untouched.

**Recognition Training and Evaluation Pipeline Refactor**:

- created `src/recognition/recognition_dataset_loader.py`
- implemented support for loading recognition data from `custom`, `orl`, `lfw`, or `all`
- implemented shared loading logic for both gallery and probe splits
- added automatic identity namespacing when training on `all` to avoid collisions across datasets
- created `src/recognition/recognition_model_trainer.py`
- implemented configurable recognition training modes: `custom`, `orl`, `lfw`, `all`
- kept a single recognition model output in `models/lbph_model.xml` and `models/label_map.pkl`
- created `src/recognition/recognition_model_evaluator.py`
- implemented configurable recognition evaluation modes: `custom`, `orl`, `lfw`, `all`
- created `src/recognition/recognition_experiment_runner.py`
- aligned the recognition experiment flow with the PAD structure by separating loader, trainer, evaluator, and experiment runner
- supported independent train / eval mode selection in the recognition runner

This replaced the previous single-dataset recognition workflow with a modular experiment pipeline aligned with the PAD architecture.

**Recognition Debug Logging and Experiment Tracking**:

- added lightweight progress logging to the new recognition loader
- added gallery / probe loading summaries with identity counts, sample counts, and timings
- added timing information for dataset loading, LBPH training, evaluation, and total experiment runtime
- added periodic evaluation progress reporting for large probe sets
- created and organized `recognition.txt` as a compact recognition experiment note file
- summarized the first `all -> all` recognition experiment inside `recognition.txt`

Initial observations from the first combined recognition run:

- `ORL` evaluation remained strong
- `custom` currently acts mainly as a sanity check because it is still small
- `LFW` evaluation is much harder for the current LBPH setup
- the main runtime bottleneck is LFW evaluation rather than training

**Recognition Legacy Cleanup**:

- removed the legacy recognition loader from `src/dataset/recognition_dataset_loader.py`
- removed the legacy recognition scripts `src/recognition/train_model.py` and `src/recognition/test_model.py`
- kept the new modular recognition pipeline as the single active path for recognition experiments

**PAD Feature Extension and Debug Alignment**:

- created `src/antispoofing/pad_lftp_features.py`
- implemented a first lightweight LFTP descriptor based on local FFT-magnitude texture codes and a normalized 256-bin histogram
- updated `src/antispoofing/pad_tgsd_features.py` to replace the previous single-scalar version with a cleaner TGS-map-based formulation
- extended TGSD extraction to two scales through the new `extract_tgsd_features()` interface
- updated `src/antispoofing/pad_feature_extractor.py` so the full PAD vector now includes:

  - original-scale base features
  - original-scale TGSD
  - downsampled-scale base features
  - downsampled-scale TGSD
  - original and downsampled LFTP histograms

- updated the PAD debug feature selection flow so TGSD and LFTP stay isolated as explicit feature blocks
- updated `src/antispoofing/training/pad_feature_extractor_tester.py` to validate the new LFTP block

This extends the PAD feature space with additional frequency-based information while keeping the feature families separated enough for future ablation work.

**PAD Debug Logging Alignment**:

- added dataset loading summaries to `src/antispoofing/training/pad_dataset_loader.py`
- added progress logging with larger step intervals suitable for PAD dataset sizes
- added timing output for PAD dataset loading, PAD classifier training, PAD evaluation, and total PAD experiment runtime
- updated the PAD experiment runner to report the total experiment duration

**Project Tracking Cleanup**:

- updated `.gitignore` to ignore the new unified `data/recognition/` tree instead of the older `recognition_train` / `recognition_test` folders
- added `recognition.txt` to the ignored experiment-note files alongside `antispoofing.txt`

This improves observability during long PAD runs without producing excessive console output.

---

## 2026/03/31

**Recognition Runtime and Evaluation Alignment**:

- updated `src/recognition/face_recognizer.py` so the recognition runtime now loads a dedicated JSON config together with the LBPH model and label map
- added `models/recognition_config.json` as the single source for the runtime `unknown_threshold`
- updated `src/pipeline/run_pipeline_full.py` to use the loaded runtime threshold instead of relying only on a fixed constant
- added a runtime guard so recognition is skipped safely when `preprocess_face_for_recognition(...)` returns `None`
- refined `src/preprocessing/face_preprocessing.py` and related recognition comments/docstrings to stay aligned with the current runtime flow

This closed the main inconsistency between recognition evaluation and runtime inference and reduced the risk of runtime crashes on invalid face crops.

**Recognition Experiment Policy Cleanup**:

- extended `src/recognition/recognition_dataset_loader.py` with the explicit `custom_orl` mode to separate the realistic `custom + orl` training setup from the wider `all` experiment that still includes `lfw`
- updated `src/recognition/recognition_experiment_runner.py` and `src/recognition/recognition_model_trainer.py` so LBPH parameters can now be passed explicitly for controlled tuning runs
- updated `src/recognition/recognition_model_evaluator.py` to:

  - compute richer score statistics
  - report the worst-performing identities
  - analyze candidate thresholds on full evaluation records
  - distinguish the `saved_runtime_threshold` from the threshold suggested by the current run
  - allow runtime threshold updates only from defensible runs with both genuine and impostor samples

- fixed the runtime-threshold policy so the official config is no longer overwritten by easy fallback runs such as `custom -> custom`

This made the recognition experiments easier to interpret and aligned the threshold policy with the actual project goals.

**Recognition Tracking and Tuning Setup**:

- created and organized `recognition.md` as the main recognition experiment report
- documented:

  - the current recognition setup
  - the roadmap
  - the fixed base comparisons
  - the threshold policy
  - the LBPH tuning criteria and provisional ranking

- recorded the current base comparisons:

  - `custom -> custom`
  - `custom_orl -> custom`
  - `orl -> orl`
  - `custom_orl -> orl`

- started the LBPH tuning campaign by enabling controlled runs over `radius`, `neighbors`, `grid_x`, and `grid_y`

At this stage the best provisional LBPH configuration is `radius=1`, `neighbors=8`, `grid_x=6`, `grid_y=6`, while the `neighbors=16` branch is still pending.

**PAD Experiment Tracking Refresh**:

- created and organized `antispoofing.md` as the main PAD experiment report
- consolidated in one place:

  - the current PAD baseline
  - the TGSD and GGD/LFT roadmap
  - the fast-mode comparison rules
  - the standard metrics to keep across runs

- updated the PAD training/debug files so the current feature-ablation and runtime-oriented experiment flow is easier to track:

  - `src/antispoofing/pad_feature_extractor.py`
  - `src/antispoofing/pad_lft_features.py`
  - `src/antispoofing/training/pad_dataset_loader.py`
  - `src/antispoofing/training/pad_experiment_runner.py`
  - `src/antispoofing/training/pad_model_trainer.py`

This gives the project two synchronized experiment logs, one for recognition and one for anti-spoofing, so the next test campaigns can be compared without rebuilding the context each time.

---

## 2026/04/01

**Recognition Runtime Policy and Runtime Split Cleanup**:

- separated the recognition evaluation threshold from the live webcam threshold
- kept the offline threshold inside `models/recognition_config.json` only as evaluator-side reference data
- removed the dependency on `recognition_config.json` from the live runtime path
- updated `src/recognition/face_recognizer.py` so the runtime now loads only:

  - `models/lbph_model.xml`
  - `models/label_map.pkl`

- introduced an explicit live recognition threshold in the runtime pipeline
- clarified comments and docstrings so offline evaluation policy and live runtime policy are no longer mixed
- updated `src/recognition/recognition_model_evaluator.py` terminology to treat the saved threshold as an offline evaluation reference instead of a runtime control value

This cleaned up one of the main conceptual ambiguities of the recognition module: the threshold used to analyze dataset runs is no longer confused with the threshold used for the live webcam demo.

**Recognition Runtime Stabilization and Session-State Refactor**:

- added `src/recognition/recognition_runtime_state.py` as the temporal state manager for runtime recognition smoothing
- introduced short-window confirmation logic for recognition results before showing a stable identity
- refactored `src/pipeline/run_pipeline_full.py` into a lightweight state machine with:

  - `DETECT`
  - `PAD_CHECK`
  - `RECOGNITION_LOCKED`
  - `RESET`

- added `src/pipeline/runtime_session_state.py` to store the persistent runtime state shared across frames
- changed the runtime flow so PAD is used to enter the live phase, while recognition then works inside a separate locked phase
- reduced the risk that PAD continuously destabilizes recognition during live inference
- simplified the full runtime overlay and kept only compact, stable information on screen

This made the integrated webcam pipeline more coherent and much closer to a real runtime session flow instead of a frame-by-frame cascade with conflicting gates.

**Dedicated Runtime Entry Points for BS and CV**:

- replaced the old disabled placeholders with two real split runtimes:

  - `src/pipeline/run_pipeline_bs.py` for recognition-only runtime
  - `src/pipeline/run_pipeline_cv.py` for PAD-only runtime

- updated the entry points:

  - `src/main.py`
  - `src/main_bs.py`
  - `src/main_cv.py`

- aligned comments and docstrings so each entry point clearly states whether it runs:

  - the full integrated runtime
  - the recognition-only runtime
  - the PAD-only runtime

- added lightweight debug logging to the split runtimes without overloading the integrated full pipeline

This made it possible to isolate runtime behavior more cleanly and compare the recognition and PAD modules separately before recombining them in the full demo.

**PAD Model/Runtime Alignment and Artifact Management**:

- added `src/antispoofing/pad_model_config.py` to centralize PAD model metadata and avoid feature-mode mismatches between training, evaluation, and runtime
- updated the PAD runtime detector and evaluators to restore the correct PAD feature configuration from saved metadata
- added explicit runtime checks on the extracted PAD feature dimension to prevent silent model/config mismatches
- extended the PAD experiment flow so `fast` and `full` runs now save separate artifacts instead of overwriting the same files
- kept the canonical runtime artifacts as the stable target used by webcam inference, while preserving experiment-specific snapshots for comparison

This closes the previous gap between PAD experiments and live runtime usage, and makes it much safer to iterate on PAD without accidentally losing the stable runtime model.

**PAD Runtime Profiling and Feature-Cost Diagnosis**:

- instrumented the PAD-only runtime to profile the cost of the live PAD inference blocks
- extended the PAD feature extraction path with timing breakdowns for the main descriptor families
- confirmed that the dominant runtime cost is not detection, crop handling, quality checks, scaling, or SVM inference
- identified the current LFTP block as the main computational bottleneck during live PAD feature extraction
- verified at runtime that the active PAD configuration is the baseline `128`-feature setup, so the remaining performance issue comes from unnecessary feature work rather than from loading the wrong saved model

This profiling pass gave a concrete explanation for the remaining runtime lag and turned the PAD performance problem from a generic suspicion into a measurable feature-extraction bottleneck.

---

## 2026/04/02

**Recognition**:

- stabilized the `bs` runtime with a slower visual state machine, hold-time / hysteresis, cleaner labels, and less log spam
- validated live behavior on `custom:valerio`; difficult tablet probes now tend to fall back to `Unknown` instead of producing unstable identity switches
- verified that `custom:francesca` is present in the trained model and current custom split

**PAD**:

- fixed the runtime feature path so TGSD / LFTP are computed only when the active mode requires them
- removed the leftover unconditional LFTP timing block that was still causing webcam lag
- realigned PAD runtime metadata / artifacts and promoted the coherent `fast` `baseline + ggd_light` (`124`-feature) set to the canonical runtime files
- confirmed in `cv` that the old `~450ms` LFTP spike disappeared, with PAD checks now roughly in the `11-20ms` range
- completed the lightweight PAD comparisons and confirmed `baseline + TGSD + GGD(LFT)_light` as the strongest fast configuration
- tested tighter TGSD variants and kept `tau170 | sobel | top5` as the promoted setting; `prewitt`, `top10`, and `GGD(LFT)_true` remain archived as non-promoted branches
- promoted the final winning PAD config to the canonical runtime artifacts and verified in `cv` that it stays responsive while handling live/spoof transitions well
- completed the `full` PAD verification on `CASIA`, `Replay-Attack`, and runtime test splits; the winning config stayed strong on spoof/general datasets while the remaining weakness is still live recall on the custom runtime split

**Project Tracking**:

- updated `recognition.md` and `antispoofing.md` to reflect the current runtime state, the final fast PAD comparison, and the subsequent `full` verification
- reduced the active PAD batch script to the single winning `full` run with retry support
- cleaned `.gitignore` for generated model artifacts and local runtime configs

**PAD Runtime-Only Augmentation and Live-FN Analysis**:

- extended the PAD experiment flow to log runtime false negatives for the `live` class with image paths and live probabilities
- used the new logging to inspect the hardest `live` runtime failures and confirmed that the main weakness is still the real `live` side rather than spoof rejection
- added photometric live augmentation support during PAD training with a controlled `mix` policy
- compared:

  - `mix`
  - `gamma`
  - `brightness_contrast`
  - `side_shading`
  - `gamma + side_shading`

- confirmed that the broad `mix` helps more than single isolated branches
- confirmed that the most useful improvement comes from applying the photometric live augmentation only to the runtime live training subset
- promoted the current PAD fast reference to:

  - `baseline + TGSD + GGD(LFT)_light`
  - `tau170 | sobel | top5`
  - `runtime-only mix`

- rejected the following PAD branches after controlled tests:

  - `light_asymmetry`
  - `runtime_repeat=2` with runtime-only augmentation
  - alternative TGSD settings already archived in the PAD report

This closed the PAD feature-search phase and shifted the remaining focus toward live robustness and final full validation.

**PAD and Recognition Cleanup**:

- cleaned the active PAD path by removing rejected branches such as `light_asymmetry` and archiving the older non-promoted feature variants
- reduced PAD logs and aligned `run_pipeline_cv.py` with the current PAD logic used in `run_pipeline_full.py`
- reorganized `antispoofing.md` and `recognition.md` into compact reports with summary tables and updated roadmaps
- aligned recognition defaults to the current reference setup:
  - train: `custom_orl`
  - eval: `orl`
- aligned the recognition fallback runtime threshold to `45.0`
- aligned `run_pipeline_bs.py` and `run_pipeline_full.py` more closely on the recognition side by using the same frame-usability gate and by freezing recognition cleanly during the post-live-lock delay
- cleaned comments and docstrings in the active PAD and recognition modules

**Final Project-State Sweep**:

- performed a final consistency pass across PAD, recognition, and the runtime pipelines
- verified syntax correctness on the active modules
- confirmed that PAD and recognition are now both clean and aligned with the current setup
- next real integration step:
  - final `full` PAD run with the promoted `runtime-only mix` setup

---
