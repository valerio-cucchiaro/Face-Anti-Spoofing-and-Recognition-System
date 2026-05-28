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

**PAD Quality Gate and Contract Tests**:

- clarified that the shared PAD quality gate is a loader-side geometric filter and a runtime-history signal, not a duplicated detector-side check
- aligned the PAD contract tests to the actual current behavior of the post-crop path
- added the missing contract checks for raw-frame loading, final-crop loading, detector wiring, and runtime/loader consistency
- confirmed the new PAD contract test block passes cleanly
- updated the PAD roadmap to reflect the current quality-gate role and the current `full` reference as `mix` over all live samples

**PAD Runtime Tuning and Diagnostics**:

- tightened the PAD runtime so the temporal decision now waits for a full window of `5` valid samples before promoting `live` or `spoof`
- tuned the PAD runtime threshold to `enter_live = 0.65` while keeping the spoof boundary strict
- aligned `run_pipeline_full.py` and `run_pipeline_cv.py` on the PAD update cadence (`PAD_RUN_INTERVAL = 2`)
- simplified the `full` runtime logs back to the phase-flow view after the temporary PAD debug pass
- kept `cv` as the diagnostic runtime and added distance-band, spatial and lighting summaries for live PAD inspection
- confirmed in runtime tests that:
  - control light behind the face remains a hard failure mode
  - distance too close to the webcam can still collapse into `face_missing`
  - cropped faces can now unlock `LOCKED` through the crop-validity path
  - motion-triggered unlock now behaves more predictably than the old bbox-only check
- kept the final `full` lock behavior conservative but readable:
  - `face_missing = 2`
  - `movement = 1`
  - `crop = 1`

---

## 2026/04/07

**Recognition Runtime Diagnostics and Light Robustness Tests**:

- extended the recognition-only runtime `src/pipeline/run_pipeline_bs.py` with structured diagnostics for:
  - face bbox geometry
  - bbox delta against the previous frame
  - crop quality gate reasons
  - compact light bands (`good_light`, `mid_light`, `bad_light`)
- kept the recognition runtime separate from PAD while reusing the shared recognition preprocessing pipeline
- added periodic and final runtime summaries to track:
  - accepted / rejected checks
  - confidence statistics
  - light-band counts
  - overlay transitions

**Runtime Threshold Sweep on `valerio`**:

- ran controlled live recognition tests on the custom runtime with different thresholds:
  - `44.0` without strong side light: stable and permissive
  - `40.0` without strong side light: too strict
  - `41.0` without strong side light: best compromise
  - `44.0` with strong side light: strongly unstable and biased toward `Unknown`
- concluded that the live runtime threshold around `41.0` is the best compromise for the normal webcam setup

**Lighting Corruption Experiments**:

- added temporary runtime-only light compensation experiments to understand whether photometric fixes could stabilize LBPH under uneven lighting
- tested `CLAHE` on the runtime crop:
  - the logs showed that it often worsened recognition confidence
  - the experiment was discarded
- tested a light `gamma` correction on the runtime crop:
  - added per-frame diagnostics to compare `raw_confidence` vs `gamma_confidence`
  - the logs showed that gamma helped only occasionally and worsened the result more often than it improved it
  - the experiment was also discarded
- reverted the runtime back to the base recognition preprocessing plus light diagnostics only

**Recognition Reporting Update**:

- updated `recognition.md` with:
  - the live runtime threshold sweep
  - the `CLAHE` runtime comparison
  - the `gamma` runtime comparison
  - a runtime-only roadmap separated from the dataset-related work
- clarified that the remaining problem is not just brightness, but the combination of light asymmetry and crop consistency with LBPH

---

## 2026/04/09

**Custom Raw Dataset Structure**:

- switched the custom dataset flow to store raw webcam frames instead of already processed crops
- aligned PAD custom collection with `data/pad_custom_raw/train|test/{live,spoof}`
- aligned recognition custom collection with `data/recognition/custom_raw/{gallery,probe_known,probe_unknown}`
- kept detection and preprocessing inside the loaders / runtime so the dataset on disk stays closer to the original acquisition
- updated the custom dataset contract so processing happens later, not during capture

---

## 2026/04/11

**PAD and Recognition Refactor**:

- centralized the PAD crop geometry in `src/antispoofing/pad_crop_utils.py` so runtime, dataset collection, and offline evaluation share the same canonical crop bounds
- updated the PAD dataset loader to distinguish between public `raw_frame` inputs and custom `final_pad_crop` data
- extracted the official PAD metric computation into `src/antispoofing/pad_metrics.py`, including threshold sweep, EER, and report exports
- split PAD diagnostics from metric computation so the evaluator can print both human-readable logs and report-ready values cleanly
- extracted the recognition metric computation into `src/recognition/recognition_metrics.py`, including threshold sweep, EER, and report exports
- refactored `src/recognition/recognition_model_evaluator.py` to rely on the new metrics module and produce more structured diagnostics
- unified the face-bbox visualization used by `run_pipeline_bs.py`, `run_pipeline_cv.py`, and `run_pipeline_full.py` through `src/preprocessing/face_bbox_visualization.py`

---

## 2026/04/16

**Recognition Metric Expansion**:

- added an offline `all gallery vs all probe` pairwise analysis to the recognition metrics, using the same LBPH distance logic already used by the recognizer
- kept the runtime decision path unchanged:
  - `predict(...)`
  - best match
  - threshold-based `Unknown` rejection
- added pairwise/offline outputs to the recognition reports so the evaluator can now export:
  - the standard threshold / ROC / DET artifacts
  - a separate pairwise matrix report for probe-vs-gallery analysis
- derived extra offline readings from the distance matrix:
  - rank-1 / rank-k
  - best-vs-second-best margin
  - identity-level minimum distance aggregation
  - open-set summary from the pairwise view
- kept the new analysis additive so the existing recognition metrics remain valid and comparable, while the report becomes more informative for the exam / thesis discussion

**Model Artifact Refactor**:

- split the generated model artifacts into two dedicated subfolders:
  - `models/recognition/`
  - `models/pad/`
- moved the recognition artifacts to the canonical recognition folder:
  - `lbph_model.xml`
  - `label_map.pkl`
  - `recognition_config.json`
- moved the PAD runtime and experiment artifacts to the canonical PAD folder:
  - `pad_svm.joblib`
  - `pad_scaler.joblib`
  - `pad_config.json`
  - the `fast` and `full` PAD snapshots
- updated the recognition and PAD loaders / trainers / evaluators so they all resolve the new canonical paths consistently
- kept the model behavior unchanged: only the artifact layout changed, not the recognition or PAD logic

**Public Metric Batch Setup**:

- prepared the batch runner to execute only public-dataset metric checks
- recognition runs now target:
  - `custom_orl -> orl`
  - `orl -> orl`
- PAD runs now target:
  - `casia -> casia`
  - `replay -> replay`
- kept the custom datasets out of the batch for now because the custom PAD spoof split and the custom recognition unknown split are still incomplete
- kept the batch output separated into the existing recognition and PAD log files

**Recognition Report Refinement**:

- added a compact CMS/CMC curve export for recognition so the rank-based behavior can be plotted directly from the report data
- added a small pairwise preview export for a few representative probes so the distance matrix can be shown without dumping the full matrix
- kept the full all-gallery-vs-all-probe matrix export unchanged for deeper analysis
- updated the recognition diagnostics to print the CMS summary and a short preview of representative probe cases
- left the recognition runtime unchanged: the new outputs are report-only additions

---

## 2026/04/21

**Pose Metrics Layer**:

- added offline and runtime pose metrics for `roll`, `yaw`, `pitch`, and `SP` without changing the custom raw dataset collection flow
- wired the new pose summaries into recognition and PAD training / evaluation so the reports now include pose coverage, means, and extrema
- kept `roll correction` out of offline runs for now so the current metrics stay comparable with the existing saved data
- used the new logs to confirm that the recognition and PAD runs stay stable while pose values vary coherently across datasets and splits
- left the next step open for a separate `roll correction` pass after this baseline comparison

**PAD Runtime/Custom Refactor**:

- unified the PAD dataset naming around `custom` and `all`
- removed the old dataset-side `runtime` aliases and the duplicate repeat wrapper names
- renamed the custom PAD evaluator file and aligned its content and logs to the custom test flow
- kept `runtime` only where it still means live/runtime artifacts for the PAD inference pipeline
- verified that the current PAD custom batch runs still launch and export the expected models and metrics

---

## 2026/04/22

**Doddington Zoo**:

- made the Doddington zoo report assign exactly one dominant label per identity instead of overlapping categories
- kept all category scores in the report together with the gap between the best and second-best score
- exported the updated zoo analysis in the recognition reports so the dominant class and confidence gap are visible offline
- stress-tested the zoo on `custom -> custom`, `all -> all`, `orl -> orl`, and reduced-LFW runs so the new assignment stays consistent

**System Response Reliability**:

- added an offline SRR analysis on top of the recognition pairwise report
- computed multiple variants for the reliability score, including `relative_distance`, `density_ratio`, and `adaptive_density_ratio`
- calibrated the SRR threshold offline and exported a dedicated reliability CSV alongside the standard recognition metrics
- kept SRR as a report-only layer for now, separate from the runtime recognition threshold

**BS Runtime Crop Alignment**:

- aligned the BS runtime recognizer crop with the offline recognition preprocessing instead of reusing the PAD crop for LBPH input
- kept the PAD crop only for quality / gating checks inside the BS runtime
- confirmed that the runtime model artifacts are still created and loaded correctly after the crop change
- verified that the BS pipeline now recognizes the custom identities consistently again with the offline-style crop

---

## 2026/04/29

**System-Level Evaluation**:

- added `src/evaluation/system_evaluator.py` as an offline evaluator for the complete biometric system
- introduced an open-set identification protocol with PAD gating and three explicit input classes:
  - `known_live`: enrolled live users
  - `unknown_live`: non-enrolled live users / biometric impostors
  - `spoof_attack`: presentation attacks
- kept biometric impostors and presentation attacks separated so recognition errors and PAD errors are measured with the correct terminology
- reported recognition-only metrics on live probes:
  - FAR
  - FRR
  - EER
  - DIR
  - misidentification rate
- reported PAD-only metrics on live/spoof samples:
  - APCER
  - BPCER
  - ACER
- added PAD-gated recognition metrics for the final system decision:
  - `final_unknown_live_far`
  - `final_known_live_frr`
  - `spoof_attacks_blocked`
  - `spoof_attacks_passed_pad`
  - `live_samples_blocked_by_pad`
  - `live_samples_passed_pad`
- exported the system-level results to:
  - `logs/metrics/system/system_evaluation_report.json`
  - `logs/metrics/system/system_evaluation_samples.csv`
- kept `run_pipeline_full.py` unchanged so the runtime demo remains separate from the offline evaluation evidence

**Recognition Probe Prediction Export**:

- added a per-probe recognition CSV export in `src/recognition/recognition_model_evaluator.py`
- exported one readable row per probe with dataset, split, path, true identity, predicted identity, LBPH distance, threshold, acceptance flag, and final recognition decision
- used decision labels aligned with the system evaluator:
  - `ACCEPTED_CORRECT`
  - `ACCEPTED_WRONG`
  - `REJECTED_UNKNOWN`
  - `FALSE_ACCEPT`
- kept the export inside the recognition evaluator so `src/recognition/recognition_metrics.py` remains focused on metric computation

**Recognition Protocol Logs**:

- added explicit gallery/probe protocol logs to the recognition trainer and evaluator
- made the enrollment/test separation visible at run time:
  - `gallery` is used for enrollment / LBPH template training
  - `probe`, `probe_known`, and `probe_unknown` are used only for evaluation
- clarified the custom open-set meaning of `probe_unknown` as non-enrolled live identities / biometric impostors

**Evaluation Package Cleanup**:

- moved PAD evaluator code into `src/evaluation/`:
  - `src/antispoofing/training/pad_model_evaluator.py` -> `src/evaluation/pad_model_evaluator.py`
  - `src/antispoofing/pad_metrics.py` -> `src/evaluation/pad_metrics.py`
  - `src/antispoofing/training/pad_custom_evaluator.py` -> `src/evaluation/pad_custom_evaluator.py`
- moved recognition evaluator code into `src/evaluation/`:
  - `src/recognition/recognition_model_evaluator.py` -> `src/evaluation/recognition_model_evaluator.py`
  - `src/recognition/recognition_metrics.py` -> `src/evaluation/recognition_metrics.py`
- updated runner/test imports to use the new evaluation module paths
- fixed the moved custom PAD evaluator project-root path after relocation
- kept trainers, dataset loaders, runtime modules, and preprocessing modules in their original packages

---

## 2026/04/30

**Recognition Validation/Test Protocol**:

- replaced the public recognition open-set evaluation layout with an explicit threshold-selection split:
  - `gallery`: enrolled identities used only for LBPH training
  - `validation_known`: enrolled identities used only to select the recognition threshold
  - `validation_unknown`: non-enrolled identities used only to select the recognition threshold
  - `test_known`: enrolled identities used only for final metrics
  - `test_unknown`: non-enrolled identities used only for final metrics
- kept the split rule consistent across datasets:
  - known identities: split by image
  - unknown identities: split by subject
  - gallery: always separate from validation and test
- ORL structure:
  - `gallery`: 30 known subjects, images `1-5`
  - `validation_known`: same known subjects, images `6-7`
  - `test_known`: same known subjects, images `8-10`
  - `validation_unknown`: 5 non-enrolled subjects, all images
  - `test_unknown`: 5 non-enrolled subjects, all images
- LFW structure:
  - `gallery`
  - `validation_known` / `test_known`
  - `validation_unknown` / `test_unknown`
  - unknown subjects are split between validation and test, not mixed by image
- Multi-PIE structure:
  - `gallery`
  - HR: `validation_known_hr`, `validation_unknown_hr`, `test_known_hr`, `test_unknown_hr`
  - LR: `validation_known_lr`, `validation_unknown_lr`, `test_known_lr`, `test_unknown_lr`
  - unknown HR/LR subjects are split by subject between validation and test
- kept custom recognition data on the live-style `gallery`, `probe_known`, `probe_unknown` layout for runtime/system samples
- updated the recognition evaluator so the official threshold is selected on validation and then frozen for the final test metrics
- renamed open-set metric exports to make the final evaluation scope explicit:
  - `*_test_predictions.csv`
  - `*_all_gallery_vs_all_test_matrix.csv`
  - `*_all_gallery_vs_all_test_cms_points.csv`
  - `*_all_gallery_vs_all_test_preview.csv`
- added `threshold_source` and `evaluation_scope` to the recognition metric reports and threshold-summary CSVs

---

## 2026/05/01

**Recognition Decision Logic**:

- added a dedicated recognition decision layer for identity-level candidate ranking
- separated decision logic, threshold calibration, and metric computation
- added validation-based calibration for distance and margin thresholds
- added reliability-gated offline metrics using `distance + margin` as a security-oriented comparison against the distance-only baseline
- kept the older adaptive-density SRR report as diagnostic-only evidence, while using margin as the practical reliability gate

**Recognition Experiments**:

- updated the batch workflow to run focused recognition experiments without deleting the commented alternatives
- checked custom-to-custom, ORL-to-ORL, and Multi-PIE-to-Multi-PIE behavior with the new offline decision reporting
- documented that Multi-PIE is a stress/failure case for the current LBPH setup, especially in LR probes

**Documentation**:

- updated `recognition.md` to clarify that offline recognition thresholds are selected on each dataset validation split and then frozen on the corresponding test split
- added the current distance-only vs distance+margin interpretation for the latest offline recognition reports

**Runtime Notes**:

- enabled and inspected SRR-style runtime diagnostics in the BS recognition pipeline
- corrected the SRR candidate interpretation to aggregate by identity rather than by duplicate templates of the same identity
- briefly tested a shared third-party log silencing helper, then removed it because MediaPipe/TFLite native stderr warnings were not reliably suppressed

---

## 2026/05/06

**Recognition Quality Filter**:

- added `src/recognition/recognition_quality_filter.py` for recognition-sample quality diagnostics
- added quality modes:
  - `none`: disables quality reports and filtering
  - `report_only`: records quality diagnostics without filtering samples
  - `probe_only`: keeps the gallery unchanged and rejects only invalid validation/test probes
- exported compact quality reports:
  - `recognition_quality_report.csv`: per-sample quality values and decisions
  - `recognition_quality_split_summary.csv`: per-split kept/rejected counts, identity coverage, and reject reasons
  - `quality_debug_values.csv`: values for saved debug previews
  - `quality_threshold_summary.txt`: active quality thresholds
- saved debug previews under `logs/debug/recognition_quality/<reason>/`, capped at 20 images per category
- kept pitch, roll, and spatial-pose issues as warning-only so they do not remove otherwise usable samples

**Recognition LBPH Histogram Ensemble**:

- kept OpenCV single-radius LBPH as the official recognition pipeline (`--radius`)
- replaced the temporary multi-radius vote experiment with an explicit histogram-distance test pipeline enabled by `--lbph-histogram-ensemble`
- added `src/recognition/lbph_histogram_ensemble.py` to compare probe LBPH histograms against all gallery templates across radii, normalize each radius by median gallery distance, and rank identities by minimum template distance
- saved experimental artifacts under `models/recognition/lbph_histogram_ensemble/`, while `recognizer_config.json` switches the evaluator between `lbph_single` and `lbph_histogram_ensemble`
- current ORL and Multi-PIE checks make this useful as a controlled test pipeline, but not strong enough to replace the official LBPH baseline

**Metrics Export Policy**:

- split metric artifacts by purpose:
  - `official`: report-ready JSON/summary/prediction outputs
  - `curves`: ROC/DET points, enabled with `--export-curves`
  - `debug`: heavy or investigative exports, enabled with `--debug-recognition` or `--debug-quality`
- moved PAD exports to `logs/metrics/pad/official` and optional PAD curves to `logs/metrics/pad/curves`
- moved recognition official exports to `logs/metrics/recognition/official`, optional ROC/DET to `logs/metrics/recognition/curves`, and heavy diagnostics to `logs/metrics/recognition/debug`
- kept `recognition_quality_split_summary.csv` as official and moved per-sample quality reporting behind `--debug-quality`
- moved system-level exports to `logs/metrics/system/official`

---

## 2026/05/07

**Runtime/System Threshold Policy and Validation**:

- separated benchmark runs from runtime calibration runs: benchmark datasets export metrics only, while runtime thresholds require an explicit calibration command
- recognition metrics are exported for every dataset, but `models/recognition/recognition_config.json` is updated only with `--save-runtime-threshold`
- restricted persisted recognition threshold calibration to the custom-to-custom setup used by the offline system evaluation
- updated system-level evaluation so it consumes an already saved recognition threshold config, or a manual CLI override, instead of silently falling back to the historical LBPH threshold `44.0`
- kept PAD threshold at the uncalibrated probability default `0.5` for now and record its source as `default_probability_threshold_uncalibrated`
- ran the custom recognition calibration command with `--save-runtime-threshold` and saved:
  - `unknown_threshold=30.6934`
  - `threshold_source=custom_to_custom_validation`
  - `train_mode=custom`
  - `eval_mode=custom`
- confirmed that `system_evaluator` uses the saved recognition threshold and exports under `logs/metrics/system/official`
- latest system-level result with saved recognition threshold and PAD default `0.5`:
  - recognition-only: `FAR=0.0000`, `FRR=0.0000`, `DIR=1.0000`, `EER=0.0000`
  - PAD-only: `APCER=0.0800`, `BPCER=0.6600`, `ACER=0.3700`
  - PAD-gated recognition: `final_unknown_live_far=0.0000`, `final_known_live_frr=0.4333`, `final_known_live_dir=0.5667`
- interpretation: in the current integrated custom system, recognition is not the bottleneck; PAD blocks many live samples and dominates the final known-live FRR
- methodological note: the custom system evaluation remains runtime-like/demo-oriented because the custom probe split is used for both threshold calibration and final system evaluation

---

## 2026/05/11

**Project Refactor and Runtime Polish**:

- completed a general project organization pass, including data/log path cleanup and recognition split alias cleanup
- aligned the runtime variants around the official full pipeline: CV keeps PAD-only diagnostics, BS keeps recognition-only diagnostics
- removed PAD roll correction from the CV runtime after live webcam diagnostics showed worse live stability
- removed SRR from the BS runtime after diagnostics showed it was too strict for the small family-based custom gallery
- tuned the BS recognition runtime threshold for webcam behavior and kept SRR out of the decision path
- polished the full pipeline webcam overlay: centered bbox text, removed PAD debug values, simplified acquisition warnings, added too-close/too-far guidance, and stabilized warning-to-bbox transitions
- cleaned full-pipeline console logs for demo use: app-level prefix, startup/shutdown messages, key state transitions, non-spamming face-acquisition logs, and no transient reset overlay

---

## 2026/05/12

**Recognition Refactor Cleanup**:

- clarified the custom recognition collector contract: it saves unprocessed BGR face crops, while LBPH preprocessing remains in loader/runtime
- removed the stale experimental LBPH histogram ensemble path from trainer, runner, evaluator, source files, and model artifacts
- kept the official recognition pipeline on single OpenCV LBPH with `models/recognition/lbph_model.xml`, `label_map.pkl`, and `recognizer_config.json`
- verified a fresh `custom -> custom` recognition run after cleanup, with the active config remaining `recognizer_type=lbph_single`
- renamed the saved recognition threshold artifact to `models/recognition/system_recognition_threshold_config.json` to mark it as system-evaluation input, not webcam runtime config
- replaced evaluator use of the runtime `DEFAULT_PREDICTION_THRESHOLD` with a local offline fallback threshold while keeping the live runtime thresholds separate
- added `--save-system-threshold` as the preferred calibration flag and kept `--save-runtime-threshold` only as a legacy alias
- added inline comments around threshold constants, saved system config, and CLI aliasing to make the offline/system/runtime threshold separation explicit
- replaced the open-set distance threshold selector with the official sweep-based constrained calibration:
  - primary rule: maximize DIR among validation thresholds satisfying FAR and known-wrong-accept constraints
  - fallback rule: preserve a relative DIR floor before minimizing FAR/FRR/MisID penalty on difficult overlapping datasets
  - kept the old midpoint rule only as unavailable-data fallback
- compared midpoint, penalty-only, and constrained calibration on `custom`, `orl`, and `multipie`:
  - custom remains perfect with the constrained rule
  - ORL improves security, reaching `FAR=0.0000` with only a small DIR/FRR tradeoff
  - Multi-PIE remains intrinsically weak/overlapping, but the constrained fallback avoids the penalty-only reject-all behavior
- made reliability/margin diagnostics opt-in with `--enable-reliability-diagnostics`
- added batch-level `-EnableReliabilityDiagnostics` to forward that opt-in flag to recognition runs without changing the default batch behavior
- kept SRR and reliability-gated metrics explicitly diagnostic-only, disabled by default, and separate from the official distance-threshold decision used by offline reports/runtime
- added console impact logging for the optional gate (`blocked_false_accepts`, `blocked_misidentifications`, `lost_correct_accepts`, `extra_known_rejects`) to decide whether the diagnostic gate is useful on a dataset
- fixed combined-dataset identity handling so namespaced identities such as `custom:valerio` and `orl:s1` remain distinct in mixed gallery modes, while single-dataset modes keep the simpler normalized identity names
- verified the mixed `custom_orl -> orl` path: the label map preserves dataset prefixes, ORL probes match `orl:*` identities, and custom gallery identities no longer collide with public-dataset identities
- added `predict_identity_safe(...)` for webcam runtime recognition so recognizer failures or missing label-map entries return `Unknown` with a `failure_status` instead of crashing the session
- updated the full runtime and BS recognition runtime to use the safe prediction wrapper while leaving the original strict `predict_identity(...)` contract available for tests and offline debugging
- added focused tests for safe prediction behavior when `recognizer.predict(...)` fails or returns an unknown numeric label
- documented the current LBPH preprocessing limitation: recognition uses grayscale crop, padding, resize, and histogram equalization, but no landmark-based face alignment; therefore stability depends on MediaPipe crop consistency, pose, lighting, and quality gates

**PAD Refactor Cleanup**:

- clarified the system-level evaluation contract in `system_evaluator.py`: known/unknown live samples come from custom recognition probes so open-set recognition can be evaluated after PAD, while spoof attacks come from the custom PAD test split
- documented that both sources are custom webcam-collected data, but the system evaluation remains an offline integrated approximation rather than a frame-by-frame reproduction of the webcam temporal loop
- updated the exported methodological note so official system reports state that spoof attacks are evaluated at PAD level only and are excluded from recognition FAR
- added `AntiSpoofingDetector.predict_safe(...)` for webcam runtimes so PAD feature/scaler/classifier failures return an explicit `unknown` result with `status=predict_failed` instead of closing the session
- updated the full and CV PAD runtimes to use the safe PAD prediction wrapper and log the underlying error while keeping strict `predict(...)` available for tests and offline debugging
- added focused PAD runtime tests for safe handling of feature extraction failures and feature-dimension mismatches
- documented the PAD threshold split: offline/system metrics use a per-frame live-score threshold for APCER/BPCER/FAR/FRR, while the webcam runtime uses a separate temporal smoothing and hysteresis state machine over multiple usable frames
- kept the PAD logic unchanged and clarified only the evaluation/runtime decision boundary in comments and report documentation
- clarified that offline PAD quality gating rejects structurally invalid crops, while runtime history updates additionally require sharpness and bbox stability
- documented that runtime PAD `current_label/current_score` are latest-frame diagnostics, while `final_label` is the smoothed decision used by the pipeline
- added PAD predict/proba mismatch diagnostics to the CV runtime logs and offline PAD evaluator without changing PAD decisions
- confirmed the PAD predict/proba mismatch is negligible in the CV runtime check (`1/358`, one usable), so no PAD decision refactor is needed

**Runtime Pipeline Audit**:

- added a hidden PAD re-check during `RECOGNITION_LOCKED`: it mirrors the live-lock logic for spoof evidence and resets the session only after stable usable spoof confirmations, while keeping `too close` as demo/user-position guidance rather than spoof evidence
- added webcam read retry handling in the full, CV, and BS runtimes: transient `cap.read()` failures are tolerated, while repeated failures raise an explicit runtime error
- wrapped runtime PAD/recognition startup artifact loading with clearer fail-fast errors so missing model/config files stop the demo with actionable messages

**Dataset and Splits Audit**:

- renamed public recognition final splits from `test_*` to `probe_*` (`probe_known/probe_unknown`, plus Multi-PIE HR/LR variants) to match biometric terminology
- hardened the Multi-PIE splitter with fail-fast HR/LR completeness checks and confirmed the regenerated split is balanced (`30` known, `10` unknown, `Missing LR matches: 0`)
- added custom recognition collector guards so an enrolled gallery identity cannot be collected as `probe_unknown`, and an existing unknown cannot later be enrolled into `gallery`

---

## 2026/05/13

**Metrics and Logs Audit**:

- clarified PAD terminal logs: `classifier.predict` accuracy/confusion matrix are diagnostic hard-label metrics, while official PAD APCER/BPCER/FAR/FRR use live-score thresholding
- removed fixed PAD probability-column assumptions by reading spoof/live indices from `classifier.classes_` in runtime and offline evaluators
- clarified recognition logs/docs by presenting OpenCV LBPH `confidence` as a distance (`lower is better`) in terminal-facing output
- made PAD, recognition, and system JSON reports include their generated artifact paths so each report can serve as a complete export index
- made the PAD confusion matrix log explicit about row/column meaning and fixed its class order to `[spoof(0), live(1)]`

**Testing and Reproducibility Audit**:

- fixed PAD SVM probability reproducibility by setting `random_state=42` on `SVC(probability=True)`
- extended the recognition artifact config with training metadata (`train_mode`, LBPH params, preprocessing, quality mode, sample ratio, dataset counts, and run timestamp)
- made full PAD experiments the default runner behavior and logged whether trained artifacts are promoted to the runtime PAD paths; `fast` remains available only as an explicit CLI speed
- added a test fixture that resets recognition preprocessing globals around every test to prevent cross-test state leakage
- anchored public dataset splitters and Replay-Attack frame extraction paths to `PROJECT_ROOT` so they no longer depend on the caller's working directory
- made Replay-Attack frame extraction tolerant of OpenCV builds without `cv2.utils.logging`
- added visible START/END banners to dataset splitters/frame extraction and summarized unreadable Replay-Attack videos while suppressing repeated native FFmpeg zlib header errors
- added `.python-version` with Python `3.12.10` to formalize the environment version used by the project
- added regression tests for PAD config mismatch handling, recognition artifact metadata, complete JSON `export_paths`, and PAD full/fast runtime promotion behavior; full test suite verified with `78 passed`

---

## 2026/05/15

**PAD Public Validation Protocol**:

- added side validation evaluation modes for public PAD datasets: `replay_validation` and `casia_validation`
- kept the original public test modes (`replay`, `casia`) available as pre-validation baselines
- added Replay-Attack extraction split policy support so `raw/devel` can be mapped to `data/pad/replay_attack_pad/validation` while preserving the legacy merge behavior when requested
- added CASIA PAD validation splitter based on subject-level move from `train` to `validation`, avoiding subject leakage between train and validation
- removed the earlier CASIA video-level split option after diagnostics showed it made validation unrealistically easy
- added validation dataset loaders for Replay-Attack and CASIA PAD
- added APCER-constrained threshold selection: choose the validation threshold with `APCER <= target_apcer`, then minimize BPCER under that constraint
- set the public validation baseline target to `APCER <= 0.02` for both Replay-Attack and CASIA
- kept EER only as a reported diagnostic metric, not as the threshold selection rule
- tested Replay-Attack validation thresholds at `1%`, `2%`, `2.5%`, and `3%` APCER; selected `2%` as the baseline because it preserves a security-oriented operating point without the overly severe live rejection of `1%`
- tested CASIA first with video-level validation, then replaced it with subject-level validation after the video-level split produced perfect validation metrics and an over-strict test threshold
- selected CASIA subject-level validation with `APCER <= 2%` as the final public CASIA baseline
- final baseline protocol for public PAD datasets is now:
  - train on `train`
  - select the live-score threshold on `validation` with APCER-constrained selection at `2%`
  - evaluate once on `test` with the frozen validation threshold

---

## 2026/05/16

**Recognition Mode Cleanup**:

- removed the unused `custom_orl` recognition mode from the Python pipeline instead of keeping partial/legacy support
- updated recognition dataset mode resolution, identity namespacing, CLI train/eval choices, and trainer-facing mode descriptions so the supported set is now `custom`, `orl`, `lfw`, `multipie`, and `all`
- kept `all` as the only mixed recognition mode and rechecked docstrings/comments in the modified recognition files for consistency with the simplified mode logic
- added custom PAD pseudo-validation with deterministic internal holdout from `train`, using the held-out subset for threshold selection and keeping `test` as the final frozen evaluation split
- added custom recognition pseudo-validation with deterministic internal holdout from `probe_known` and `probe_unknown`, so validation and final probe are now separated even without dedicated custom validation folders
- cleaned legacy PAD data folders by renaming `casia_pad_backup_pre_validation` to `casia_pad_raw` as the preserved original CASIA dataset and removing the obsolete `casia_pad_video_validation_split` and `replay_attack_pad_legacy_baseline` copies
- rewrote the CASIA PAD splitter to generate `casia_pad` from immutable `casia_pad_raw`, recreating the output split from scratch, copying `test` unchanged, and verifying final counts instead of destructively moving files inside the working dataset
- fixed the LFW splitter final summary so it now reports selected vs actually copied known identities, copied unknown identities per split, total skipped identities, and aggregated skip reasons without changing the split policy
- fixed the ORL splitter final summary so it now reports only subjects and files actually copied, explicitly counts skipped identities, and avoids misleading theoretical totals when an identity is incomplete
- refreshed outdated PAD evaluator/loader comments and docstrings in the three touched files so they now match validation modes, target-APCER threshold selection, custom pseudo-validation, and the current generic PAD split loader contract
- added focused tests for PAD validation threshold selection, custom PAD/custom recognition internal validation splits, CASIA/LFW/ORL splitter behavior and summaries, and verified the full suite passes with `92 passed`
- moved custom PAD internal train/validation splitting after loading and quality filtering to align it with custom recognition; the custom PAD baseline metrics stayed unchanged on the current dataset, while training time increased because the train split is now scanned once to determine valid samples before building the deterministic holdout

**Runtime Refactor**:

- aligned the `too close` runtime branch with the post-lock face-unusable reset behavior already used for `too far` / missing-face conditions, so an oversized face now drops the live lock after the configured miss threshold instead of preserving it indefinitely
- removed unused runtime session state artifacts from `runtime_session_state.py`, including `last_valid_face_rect`, `live_lock_active`, and `last_stable_identity`, after confirming they were no longer consumed by the full runtime pipeline
- clamped MediaPipe face-detection bounding boxes to valid frame coordinates before downstream size/stability checks, skipping degenerate boxes instead of passing raw out-of-frame rectangles through the runtime
- wrapped full-pipeline MediaPipe detector initialization in a contextual runtime error so detector startup failures now match the clearer PAD and recognition initialization behavior
- normalized the locked recognition overlay so the identity line now always shows `Identity: ...` text, using `Identity: pending` during early recognition stabilization instead of leaking process-state messages into the identity slot
- refined face-acquisition terminal logs so `too close` warnings are reported explicitly instead of being logged under the generic `face_lost` reason
- renamed the shared PAD crop-history quality flag from `is_frame_usable_for_history` to `is_crop_usable_for_history` to make clear that it covers only crop geometry and sharpness, while runtime pipelines may still add bbox-stability gating on top
- clarified the live-lock PAD comments so session reset remains tied to crop validity, while spoof-based lock release keeps the stricter requirement of fully usable PAD evidence
- made PAD runtime detector failure payloads uniform across `invalid_input`, `ok`, and `predict_failed`, so webcam callers can always inspect the same keys without special-case guards
- switched the PAD runtime detector to derive the final runtime `label` from `live_score` while keeping the raw SVM `predict(...)` result as diagnostic `prediction_label`, and added regression coverage for explicit predict/proba mismatch handling

**PAD Feature Recheck**:

- saved a fresh `all -> all | full | baseline + ggd_light` PAD comparison run as the baseline branch reference before rechecking `tgsd`
- training counts for this run:
  - CASIA train valid: `12869 live / 28485 spoof`
  - Replay train valid: `2145 live / 2252 spoof`
  - combined train total: `15014 live / 30737 spoof` (`45751` samples, `feature_dim=124`)
- public test results with this branch:
  - CASIA test: `APCER=0.0202`, `BPCER=0.2558`, `ACER=0.1380`, `EER=0.1147`
  - Replay-Attack test: `APCER=0.0124`, `BPCER=0.2449`, `ACER=0.1287`, `EER=0.0599`
- custom threshold-selection and final custom test results degraded sharply with this branch:
  - custom pseudo-validation selected `threshold=0.9097`, with `APCER=0.0000`, `BPCER=0.8125`, `ACER=0.4062`
  - custom test with that frozen threshold reached `APCER=0.0000`, `BPCER=0.9500`, `ACER=0.4750`, confirming that `baseline + ggd_light` is currently too conservative on custom live samples despite acceptable public-dataset security
- saved the matching `all -> all | full | tgsd + ggd_light` comparison run:
  - training counts remained the same, with `feature_dim=126`
  - CASIA test: `APCER=0.0183`, `BPCER=0.2701`, `ACER=0.1442`, `EER=0.1141`
  - Replay-Attack test: `APCER=0.0103`, `BPCER=0.2247`, `ACER=0.1175`, `EER=0.0505`
  - custom pseudo-validation selected `threshold=0.8018`, with `APCER=0.0000`, `BPCER=0.6875`, `ACER=0.3438`
  - custom test with that frozen threshold reached `APCER=0.0000`, `BPCER=0.8375`, `ACER=0.4188`
- first comparison reading:
  - `tgsd + ggd_light` improves Replay and clearly improves custom live retention relative to `baseline + ggd_light`
  - CASIA remains close, with slightly better spoof-side security (`APCER`) but slightly worse live-side rejection (`BPCER`)
  - on the current evidence, `tgsd + ggd_light` is the stronger branch overall because the custom degradation is materially smaller while public-dataset behavior stays comparable
- saved the dedicated `custom -> custom | full | baseline + ggd_light` run:
  - custom train effective subset: `64 live / 112 spoof` (`176` samples, `feature_dim=124`)
  - custom pseudo-validation selected `threshold=0.7896`, with `APCER=0.0000`, `BPCER=0.2500`, `ACER=0.1250`
  - custom test with that frozen threshold reached `APCER=0.0100`, `BPCER=0.1500`, `ACER=0.0800`, with `FAR=0.0100`, `FRR=0.1500`, `GAR=0.8500`
- this confirms that `baseline + ggd_light` remains strong on the pure custom-only PAD branch even though it degrades badly when promoted as the shared `all -> all` full branch
- saved the dedicated `custom -> custom | full | tgsd + ggd_light` run:
  - custom train effective subset: `64 live / 112 spoof` (`176` samples, `feature_dim=126`)
  - custom pseudo-validation selected `threshold=0.8007`, with `APCER=0.0000`, `BPCER=0.2500`, `ACER=0.1250`
  - custom test with that frozen threshold reached `APCER=0.0100`, `BPCER=0.1625`, `ACER=0.0863`, with `FAR=0.0100`, `FRR=0.1625`, `GAR=0.8375`
- promoted `tgsd + ggd_light` to the project default PAD feature branch:
  - it stays the stronger overall branch for the shared `all -> all` full configuration
  - `baseline + ggd_light` remains documented as the slightly stronger custom-only comparison branch, but no longer as the default
- aligned the hardcoded PAD feature default in `pad_feature_extractor.py` with the already promoted runtime artifacts so the code default, saved runtime config, and current documentation now all point to `tgsd + ggd_light`
- removed the obsolete standalone `src/evaluation/pad_custom_evaluator.py` after confirming it was no longer referenced by active evaluation flows or tests; the maintained custom PAD protocol now lives entirely inside `pad_model_evaluator.py` through `evaluate_custom_pad_protocol(...)`

---

## 2026/05/17

**PAD Runtime Hardening**:

- added an explicit fail-fast guard in `pad_runtime_detector.py` so runtime PAD artifact loading now rejects classifiers without `predict_proba(...)` before inference starts
- documented that this check protects the probability-driven runtime contract, where `live_score` is required both for the returned PAD payload and for the smoothed webcam decision logic

**Recognition Threshold Calibration Hardening**:

- removed the offline recognition threshold fallback paths that previously invented threshold values when calibration data was incomplete
- `compute_suggested_threshold(...)` and `compute_open_set_threshold(...)` now return no threshold instead of using fixed `+5` margin fallbacks without impostor/unknown evidence
- custom open-set recognition evaluation now fails explicitly when validation threshold calibration cannot derive a valid threshold from the observed score distributions
- this keeps threshold-based offline recognition reports aligned with real calibration evidence instead of silently continuing with arbitrary fallback numbers
- cleaned the offline recognition threshold diagnostics so open-set and closed-set reports no longer print legacy `conservative_threshold` or `offline_fallback_reference` candidates after the fallback policy was removed
- restricted the canonical runtime recognition artifacts to the `custom` train mode, while non-custom recognition runs now save isolated mode-specific snapshots instead of overwriting the live runtime model
- made the runtime recognition loader validate `recognizer_config.json` for the canonical custom artifacts, so webcam pipelines now fail fast if the active recognition model was not trained from the expected custom branch
- stored the recognizer metadata snapshot inside `system_recognition_threshold_config.json` and made the offline system evaluator verify that the saved custom threshold matches the active canonical custom recognition artifacts before running integrated PAD+recognition reports
- fixed the first follow-up integration issues after that refactor by passing the new `recognizer_config` snapshot through every open-set evaluation branch (`orl`, `lfw`, `multipie`) and by aligning the offline system evaluator with the current custom pseudo-validation split names (`probe_known_internal` / `probe_unknown_internal`)
- verified the refactor end-to-end:
  - `custom -> custom --save-system-threshold` completed successfully and refreshed the canonical runtime recognition artifacts plus the saved offline system threshold config
  - `orl -> orl` completed successfully and wrote only isolated `_orl` artifacts, confirming that non-custom runs no longer overwrite the runtime custom recognizer
  - `system_evaluator` completed successfully against the canonical custom recognizer and the refreshed custom threshold config
- saved the resulting offline system metrics:
  - recognition-only: `threshold=31.7896`, `DIR=1.0000`, `FAR=0.1667`, `FRR=0.0000`, `misidentification_rate=0.0000`
  - PAD-only: `threshold=0.5000`, `APCER=0.0600`, `BPCER=0.2286`, `ACER=0.1443`
  - PAD-gated recognition: `final_unknown_live_far=0.0000`, `final_known_live_frr=0.0870`, `final_known_live_dir=0.9130`
  - PAD gate counts: `spoof_blocked=94`, `spoof_passed=6`, `live_blocked=8`, `live_passed=27`

**PAD Runtime Artifact Protection**:

- aligned PAD runtime artifact promotion with the same protection strategy adopted for recognition, but using the PAD project reference run instead of a single train mode
- the canonical webcam PAD artifacts under `models/pad/` are now updated only by the approved `all -> all | full` experiment
- all other PAD runs still save their own `full` or `fast` snapshots, but they no longer overwrite `pad_svm.joblib`, `pad_scaler.joblib`, or `pad_config.json`
- refreshed the related PAD runner/trainer/config docstrings and logs so they now describe the runtime-vs-snapshot distinction explicitly
2026/05/17 (test compatibility follow-up)
- Restored backward-compatible recognition evaluator dispatch by making `evaluate_single_recognition_dataset(..., recognizer_config=None)` accept legacy callers while preserving the new artifact-coherence flow.
- Restored backward-compatible recognition trainer test hooks by reintroducing custom-only `MODEL_PATH`, `LABEL_MAP_PATH`, and `RECOGNIZER_CONFIG_PATH` aliases; non-custom modes still resolve isolated artifact snapshots dynamically.
- Updated PAD experiment runner tests to match the new runtime-promotion policy:
  only `all -> all | full` promotes canonical runtime PAD artifacts, while `custom -> custom | full` writes only `*_full` experiment artifacts.

**Sibling pipeline cleanup**:

- Cleaned small diagnostic leftovers from `run_pipeline_bs.py` by removing unused `RecognitionDisplayState` fields that no longer affected runtime logic or emitted logs.
- Cleaned `run_pipeline_cv.py` by removing unused helper functions and an unused feature-mode variable that had remained from older PAD diagnostics.
- Kept the commented PAD diagnostic logs in `run_pipeline_cv.py`, but clarified in comments what they report:
  startup artifact/configuration inspection and optional final distance-band summaries.

**PAD Photometric Augmentation Recheck**:

- reran the custom-only PAD branch with photometric augmentation enabled (`mix`, `runtime_only`) and confirmed that the final custom test metrics stayed effectively unchanged relative to the current no-augmentation baseline:
  - custom test remained `APCER=0.0100`, `BPCER=0.1625`, `ACER=0.0863`
  - the selected threshold moved, but the operational result did not improve
- reran the canonical `all -> all | full` PAD branch with photometric augmentation enabled:
  - `runtime_only` did not materially change the public-dataset or custom results because augmentation was not effectively applied to the combined public training branch
  - `scope=all` produced a real tradeoff:
    - CASIA: `ACER` improved from `0.1442` to `0.1222`, but `APCER` worsened from `0.0183` to `0.0276`
    - Replay: `ACER` improved from `0.1175` to `0.0789`, but `APCER` worsened from `0.0103` to `0.0179`
    - Custom: `ACER` improved from `0.4188` to `0.3862`, but `APCER` worsened from `0.0000` to `0.0600`
- final decision:
  - keep photometric augmentation as an explicit PAD test/ablation capability
  - do not promote it to the PAD baseline
  - the current operational PAD baseline remains the no-augmentation branch (`augment_live_photometric = False`)

**Refactor By Directory**:

- started a directory-by-directory audit/refactor pass using the staged workflow:
  - pass 1: file mapping and role classification
  - pass 2: internal coherence
  - pass 3: cross-directory coherence
  - pass 4: severity classification
  - pass 5: targeted fixes only after the checklist
- completed the first full pass on `src/pipeline/`:
  - mapped the active runtime files:
    - `run_pipeline_full.py` as the canonical online pipeline
    - `runtime_session_state.py` as runtime support state
    - `run_pipeline_cv.py` as the active PAD-only runtime
    - `run_pipeline_bs.py` as the active recognition-only runtime
  - confirmed there are no legacy or probably unused files in that directory
  - confirmed the runtime wiring is coherent across:
    - pipeline -> PAD runtime artifacts
    - pipeline -> recognition runtime artifacts
    - pipeline -> shared preprocessing helpers
  - kept the intentional recognition threshold divergence between `full` and `bs` as-is because it is still used as a local test setting
  - kept the remaining commented diagnostics in `run_pipeline_cv.py` and the heavier recognition diagnostics in `run_pipeline_bs.py` as intentional observability tools
- applied one small cleanup fix after the pipeline audit:
  - removed the duplicated default-session payload definitions in `runtime_session_state.py`
  - extracted shared helpers so `__init__` and `reset_session()` now reuse the same default PAD and recognition result payloads
- completed the audit/refactor pass on `src/evaluation/`:
  - confirmed the evaluator split is now coherent:
    - `recognition_model_evaluator.py` for offline recognition
    - `pad_model_evaluator.py` for offline PAD
    - `system_evaluator.py` for integrated PAD-gated recognition
    - metrics/calibration helpers kept separate in `recognition_metrics.py`, `pad_metrics.py`, and `recognition_threshold_calibration.py`
  - aligned threshold policy across evaluators:
    - removed implicit fallback thresholds from evaluation paths
    - evaluators now fail explicitly when a required saved threshold is missing or incomplete
  - hardened artifact/threshold consistency:
    - `system_evaluator` now verifies that saved recognition/PAD threshold configs match the active canonical runtime artifacts before running
    - PAD direct evaluation now requires a saved system PAD threshold config aligned with the active PAD artifact config
    - recognition direct evaluation now requires a saved system recognition threshold config aligned with the active recognizer config
  - clarified the recognition evaluator docstring so it now documents the real artifact policy:
    - `custom` owns the canonical runtime recognizer artifacts
    - non-custom modes write isolated offline snapshots
  - aligned PAD system-threshold policy with PAD runtime-artifact policy:
    - `--save-system-threshold` is now accepted only for the canonical `all -> all | full` PAD run
    - `custom -> custom --save-system-threshold` is now rejected explicitly because it would save a threshold incompatible with the canonical PAD runtime artifacts
  - added and updated tests covering:
    - PAD threshold-selection fail-fast behavior
    - PAD/recognition saved-threshold resolution
    - PAD experiment-runner restriction for `--save-system-threshold`
- verified the new evaluation policy manually:
  - targeted tests passed:
    - `tests/test_pad_validation_protocol.py`
    - `tests/test_evaluation_export_paths.py`
    - `tests/test_custom_recognition_evaluator_dispatch.py`
  - `system_evaluator` correctly failed fast when `system_pad_threshold_config.json` was missing
  - after rerunning the canonical PAD branch with `all -> all | full --save-system-threshold`, `system_evaluator` ran successfully with aligned artifacts/configs
- current conclusion on `system_evaluator`:
  - the refactor/policy is now correct, but the integrated offline system evaluation currently performs badly on live samples
  - latest integrated result:
    - threshold sources:
      - recognition=`custom_to_custom_validation`
      - pad=`custom_internal_validation_target_apcer`
    - recognition-only remained strong:
      - `threshold=31.7896`, `DIR=1.0000`, `FRR=0.0000`, `misidentification_rate=0.0000`
    - PAD-only remained extremely strict on custom live samples:
      - `threshold=0.8018`, `APCER=0.0000`, `BPCER=0.8571`, `ACER=0.4286`
    - final PAD-gated system result was therefore poor on live traffic:
      - `final_known_live_frr=0.8261`
      - `final_known_live_dir=0.1739`
      - `spoof_blocked=100`, `spoof_passed=0`, `live_blocked=30`, `live_passed=5`
  - practical reading:
    - spoof-side security is very strong
    - live-side usability is currently unacceptable
    - so the evaluation refactor is complete, but the present system-level offline PAD-gated result should be treated as bad / not yet acceptable

## 2026/05/18

**Resumed Previous Day Refactor**:

- resumed the directory-by-directory refactor started the previous day, focusing on the active `src/antispoofing/` contract and deliberately leaving `system_evaluator` policy/work for a later pass
- removed the unused legacy PAD descriptor module:
  - deleted `src/antispoofing/pad_lftp_features.py`
  - confirmed it was no longer imported by the active runtime, training, or evaluation paths
- hardened PAD artifact loading so runtime and offline evaluators now require the explicit saved PAD config chosen by the project:
  - removed the legacy `feature_dim -> config` inference path from `pad_model_config.py`
  - missing `pad_config*.json` files now raise an explicit failure instead of silently reconstructing a legacy config
  - config/artifact mismatches still fail explicitly, but the error path is now stricter and simpler
- cleaned the remaining local references to the removed LFTP archive path:
  - updated `pad_feature_extractor.py` comments so the active feature stack now documents only the currently supported paths
  - kept `PAD_LFT_MODE` unchanged and intentionally frozen to `ggd_light`
- updated the PAD config tests to match the new strict policy:
  - removed the old test that validated legacy config inference from `feature_dim`
  - added a fail-fast test that requires an explicit saved PAD config file
- practical effect of this pass:
  - PAD runtime and PAD offline evaluation now depend only on the explicit artifact/config sets we promote
  - legacy silent recovery for missing PAD config metadata has been removed

**PAD Combined Protocol Fix**:

- fixed the methodological bug in the PAD `all -> all` evaluation flow:
  - previously the run trained on all datasets but still selected the final PAD threshold only from the custom pseudo-validation subset
  - that custom-only threshold was then reused on `CASIA test`, `Replay-Attack test`, and `Custom test`, so the old `all` protocol was not a true combined validation/test protocol
- simplified the public PAD evaluation modes:
  - removed the old direct public modes that evaluated `casia` / `replay` with an already-saved system threshold
  - promoted the validation-backed protocols to the main names:
    - `casia` now means `CASIA validation -> CASIA test`
    - `replay` now means `Replay-Attack validation -> Replay-Attack test`
- rewrote `all -> all` as a real combined protocol:
  - training still uses `CASIA train + Replay train + Custom train`
  - threshold selection now uses one global validation pool built from:
    - `CASIA validation`
    - `Replay-Attack validation`
    - `Custom pseudo-validation`
  - the selected threshold is then reused unchanged for:
    - `CASIA test`
    - `Replay-Attack test`
    - `Custom test`
- updated the PAD runner/evaluator documentation and CLI so the new semantics are explicit and the removed public direct modes are no longer exposed
- added focused regression coverage for the combined-threshold behavior in `tests/test_pad_validation_protocol.py`

**PAD Canonical Run Recheck**:

- reran the canonical `all -> all | full --save-system-threshold` PAD experiment after the protocol fix
- training counts for the new canonical run:
  - `CASIA train`: `12869 live / 28485 spoof`
  - `Replay-Attack train`: `2145 live / 2252 spoof`
  - `Custom train (base)`: `80 live / 140 spoof`
  - `Custom train (effective)`: `2200` samples through `custom_repeat=10`
  - final all-train set: `47951` samples total (`15814 live / 32137 spoof`, `feature_dim=126`)
- the new combined validation protocol selected a much lower and more balanced global threshold:
  - `threshold=0.5453`
  - `APCER=0.0199`
  - `BPCER=0.2587`
  - `ACER=0.1393`
- final test metrics with the new frozen combined threshold:
  - `CASIA test`: `APCER=0.0214`, `BPCER=0.2637`, `ACER=0.1426`, `EER=0.1172`
  - `Replay-Attack test`: `APCER=0.0078`, `BPCER=0.2163`, `ACER=0.1120`, `EER=0.0508`
  - `Custom test`: `APCER=0.0500`, `BPCER=0.2750`, `ACER=0.1625`, `EER=0.0887`
- practical reading:
  - this is a major improvement over the previous custom-driven `all` threshold (`0.8018`)
  - custom live retention is now far less severe, while public-dataset security remains comparable
  - the canonical PAD runtime artifacts and the saved offline system PAD threshold config are now both aligned with a genuinely combined operating point

**System Evaluator Recheck**:

- reran `system_evaluator` after refreshing the canonical PAD artifacts and the new combined system PAD threshold config
- threshold sources are now:
  - recognition=`custom_to_custom_validation`
  - pad=`all_combined_validation_target_apcer`
- recognition-only remained unchanged and strong:
  - `threshold=31.7896`
  - `DIR=1.0000`
  - `FAR=0.1667`
  - `FRR=0.0000`
  - `misidentification_rate=0.0000`
- PAD-only improved substantially relative to the previous strict custom-driven threshold:
  - `threshold=0.5453`
  - `APCER=0.0500`
  - `BPCER=0.3143`
  - `ACER=0.1821`
- final PAD-gated system result is now much more realistic:
  - `final_unknown_live_far=0.0000`
  - `final_known_live_frr=0.2609`
  - `final_known_live_dir=0.7391`
  - `spoof_blocked=95`
  - `spoof_passed=5`
  - `live_blocked=11`
  - `live_passed=24`
- practical reading:
  - spoof-side security is still good, though no longer maximally rigid
  - live-side usability improved dramatically relative to the previous `live_passed=5 / live_blocked=30` result
  - the integrated offline system evaluation is now back in a credible operating region for comparison with the runtime demo

## 2026/05/19

**Resumed Refactor Continuation**:

- resumed the same refactor pass on `src/antispoofing/`, restarting the inspection from the beginning to re-check the active PAD contract after the recent protocol and runtime fixes
- repeated the staged review on the antispoofing area:
  - remapped the active files and responsibilities
  - rechecked internal coherence of runtime detector, artifact/config contract, feature extraction, quality gating, and training loader behavior
  - rechecked cross-directory alignment between:
    - `pipeline -> antispoofing`
    - `evaluation -> antispoofing/models`
    - `dataset -> antispoofing`
    - `tests -> real PAD contract`
- confirmed that the main refactor-critical PAD problems are now resolved:
  - runtime PAD and offline PAD evaluation now rely on the intended canonical artifacts/configs
  - the combined PAD evaluation protocol remains aligned with the new `all -> all` semantics
  - no remaining evidence was found of the old legacy config recovery path or of the removed LFTP branch in the active flow
- clarified an important semantic point on the custom PAD dataset contract:
  - `final_pad_crop` does **not** mean “already fully preprocessed for PAD”
  - it means “already localized face/PAD crop saved on disk”
  - the offline loader still re-applies the canonical post-crop PAD preprocessing path before feature extraction
  - this keeps the custom PAD dataset consistent with the project intent: saved crops, not precomputed PAD features
- verified that the custom PAD collector and loader remain aligned:
  - the collector saves cropped BGR face regions into `data/pad/pad_custom_raw/...`
  - the loader then converts/rechecks/preprocesses those crops through the shared PAD inference/training path
- identified one genuinely removable leftover in the loader:
  - `load_pad_test_dataset_combined(...)` in `src/antispoofing/training/pad_dataset_loader.py` had no remaining call sites after the protocol redesign
  - it was therefore treated as an orphan helper from an older public-combined PAD evaluation flow
- applied the agreed minimal cleanup, still leaving the large loader split for a later dedicated pass:
  - removed the unused `load_pad_test_dataset_combined(...)` helper
  - renamed the internal PAD feature-mode naming in `pad_feature_extractor.py` so the code no longer treats the active model feature mode as if it were only a debug switch
  - kept external config/artifact compatibility unchanged by preserving the saved config key `pad_debug_feature_mode`
  - corrected the loader wording around `final_pad_crop` so it now describes “already localized crop saved on disk” rather than implying “already preprocessed”
- practical reading:
  - the `src/antispoofing/` refactor is now structurally much cleaner and more explicit
  - the remaining debt in this area is mostly non-blocking cleanup, with the eventual dataset-loader split intentionally postponed to a later pass

**PAD Loader Split**:

- completed the first structural split of the old monolithic PAD dataset loader without changing the external PAD dataset contract
- separated the internals into:
  - `pad_dataset_splits.py` for dataset paths, sampling, and deterministic custom validation-split logic
  - `pad_dataset_preprocessing.py` for augmentation and image-to-feature dispatch
  - `pad_dataset_builders.py` for train/test/validation dataset builders
- kept `pad_dataset_loader.py` as a thin compatibility facade so the public loader API remains stable during the transition
- updated the real attachment points that depended on the old loader:
  - PAD trainer
  - PAD evaluator
  - system evaluator
  - PAD loader contract tests
  - custom PAD validation-split test
- practical reading:
  - the loader is now divided by responsibility without exploding the number of files
  - future cleanup can happen inside the focused modules without having to revisit one large all-purpose loader again

**PAD All-Train Rebalance And Runtime Recheck**:

- rebalanced the canonical `train-mode all | full` PAD training mix to reduce the dominance of CASIA and give the custom webcam domain a real weight in the final model
- new `all` defaults used for the canonical PAD branch:
  - `casia_sample_ratio = 0.25`
  - `replay_sample_ratio = 1.0`
  - `custom_repeat = 20`
- resulting training mix after valid-sample extraction:
  - `CASIA train = 10360` (`54.08%`)
  - `Replay-Attack train = 4397` (`22.95%`)
  - `Custom train effective = 4400` (`22.97%`)
  - final training set = `19157` samples (`6980 live / 12177 spoof`)
- reran the canonical `all -> all | full --save-system-threshold` PAD experiment with the new mix
- new combined validation operating point:
  - `threshold = 0.6140`
  - `APCER = 0.0200`
  - `BPCER = 0.2729`
  - `ACER = 0.1464`
- final test metrics after the rebalance:
  - `CASIA test`: `APCER=0.0290`, `BPCER=0.2595`, `ACER=0.1443`
  - `Replay-Attack test`: `APCER=0.0078`, `BPCER=0.2098`, `ACER=0.1088`
  - `Custom test`: `APCER=0.0600`, `BPCER=0.2500`, `ACER=0.1550`
- practical reading of the offline rerun:
  - the dataset rebalance worked as intended
  - public-dataset metrics stayed close to the previous baseline
  - custom live retention improved slightly, at the cost of a small spoof-side worsening on custom
- reran `system_evaluator` with the rebalanced canonical PAD artifacts:
  - `PAD-only`: `threshold=0.6140`, `APCER=0.0600`, `BPCER=0.3429`, `ACER=0.2014`
  - `PAD gate counts`: `spoof_blocked=94`, `spoof_passed=6`, `live_blocked=12`, `live_passed=23`
  - `PAD-gated recognition`: `final_known_live_dir=0.6522`, `final_known_live_frr=0.3478`
- practical reading of the integrated system rerun:
  - relative to the previous combined-threshold baseline, the change is slightly worse offline
  - but the loss is limited and was accepted as a reasonable tradeoff before checking the webcam runtime again
- reran `src.main_cv` with extra PAD diagnostics enabled:
  - startup logs now report the active feature set, PAD config, runtime temporal thresholds, and final distance-band summary
  - per-check logs now also report the usable-frame PAD history window
- runtime reading after the rebalance:
  - the PAD runtime is no longer stuck in a constant spoof state
  - live and spoof states both appear, but the runtime remains unstable
  - the dominant issue still appears to be lighting, especially on `far` face crops with strong dark ratio / vertical illumination asymmetry
  - current practical decision: keep the rebalance for now and treat lighting robustness as the main remaining runtime issue

## 2026/05/26

**Recognition Report Naming And Threshold Cleanup**:

- continued the recognition-side refactor with a small focused cleanup instead of splitting the loader yet
- aligned recognition metric export naming with an artifact-like stem so offline reports can be separated cleanly when needed
- introduced a shared recognition export-stem builder in the evaluator and reused it across:
  - closed-set probe evaluation
  - custom/public open-set evaluation
  - recognition prediction CSV exports
- extended the recognition quality-report path flow so quality diagnostics now follow the same run-specific stem instead of always writing into one shared global filename
- propagated stem-aware quality exports through:
  - recognition trainer gallery loading
  - recognition evaluator validation loading
  - recognition evaluator final probe loading
- practical effect:
  - recognition metric JSON/CSV exports are now easier to preserve run-by-run
  - quality debug/summary exports no longer have to collapse into one ambiguous shared report path

**Recognition Alias And Offline Threshold Semantics**:

- removed the old CLI alias `--save-runtime-threshold` from the recognition experiment runner
- kept only the explicit `--save-system-threshold` name, which matches the real consumer:
  - offline system evaluation
  - not the webcam runtime
- rechecked recognition path aliases and confirmed the remaining custom path constants are still active compatibility shorthands rather than dead code:
  - `MODEL_PATH`
  - `LABEL_MAP_PATH`
  - `RECOGNIZER_CONFIG_PATH`
- these were intentionally left in place for now because they are still used by:
  - the canonical custom training/runtime branch
  - existing tests that monkeypatch the custom artifact paths directly

**Recognition Offline Preview Clarification**:

- removed the hardcoded offline preview threshold constant that could be misread as an evaluation fallback
- open-set validation preview scoring now runs with no accept/reject threshold and explicitly marks preview rows as:
  - `PREVIEW_ONLY`
- official offline evaluation remains strict:
  - final thresholded evaluation still requires a saved calibrated threshold config
  - if the saved threshold/config is missing or mismatched, the offline evaluator still raises an error instead of inventing a fallback
- practical reading:
  - runtime still owns its own fallback threshold behavior inside the live recognizer helper
  - offline recognition evaluation is now semantically cleaner and more obviously strict

**Recognition Refactor Status**:

- `src/recognition/` is now cleaner on export policy and threshold semantics
- the remaining structural debt is still concentrated in `recognition_dataset_loader.py`
- that larger loader split is intentionally postponed to a later dedicated pass

**Preprocessing Shared Face-Area Threshold Cleanup**:

- completed the small preprocessing pass-5 fix that was worth doing now
- introduced a shared face-validation constant in:
  - `src/preprocessing/frame_preprocessing.py`
- centralised the minimum valid face area used by:
  - `run_pipeline_full.py`
  - `run_pipeline_bs.py`
  - `run_pipeline_cv.py`
  - `recognition_dataset_collector.py`
  - `pad_runtime_dataset_collector.py`
  - `pad_dataset_preprocessing.py`
  - `recognition_dataset_loader.py`
- kept the runtime/data/training behaviour unchanged:
  - same threshold value `8000`
  - same `get_valid_face(...)` contract
  - no change to messages or per-module policy
- practical effect:
  - removes silent drift risk between runtime, collectors, and offline loaders
  - keeps the fix limited to one shared source of truth without redesigning preprocessing config

**Dataset Contract Clarifications**:

- closed the `src/dataset/` pass with documentation-only cleanup rather than logic changes
- clarified that the custom recognition splits:
  - `validation_known_internal`
  - `probe_known_internal`
  - `validation_unknown_internal`
  - `probe_unknown_internal`
  are virtual internal subsets derived from the on-disk `probe_known` / `probe_unknown` folders
- clarified that Replay-Attack extraction still supports both:
  - `validation`
  - `legacy`
  split policies, but the current PAD stack supports the explicit validation layout directly
- practical reading:
  - dataset producers and loaders were already coherent
  - the pass mainly removed ambiguity in the documented contract

**Tests Coverage And Contract Alignment**:

- closed the `tests/` refactor pass by extending coverage only where the previous review had found real gaps
- added focused helper-level tests for the runtime pipeline layer instead of trying to test the webcam loops directly:
  - `RuntimeSessionState`
  - PAD overlay state helpers in `run_pipeline_full.py` and `run_pipeline_cv.py`
  - recognition overlay state helper in `run_pipeline_bs.py`
- added explicit coverage for the recognition quality-export naming policy introduced in the recent recognition refactor:
  - quality report CSV stem-aware naming
  - quality split-summary CSV stem-aware naming
  - quality threshold-summary TXT stem-aware naming
  - quality debug-values CSV stem-aware naming
- added coverage for the previously uncovered dataset/manual utility layer:
  - `replayattack_frame_extractor.py`
  - `pad_runtime_dataset_collector.py`
  - `recognition_dataset_collector.py`
  - `multipie_dataset_splitter.py`
- tightened test isolation around the recognition preprocessing global state:
  - updated `tests/conftest.py` so tests now save and restore the active recognition preprocessing config instead of blindly forcing a hardcoded default
- aligned older PAD threshold/export tests with the current evaluator contract:
  - updated the affected test mocks so they now accept the current `artifact_tag` argument used by the real saved-threshold helpers
- practical reading:
  - the suite now covers the main refactor-sensitive contracts more explicitly without adding fragile webcam end-to-end tests
  - the new tests stay at helper / contract level and do not touch real runtime artifacts
  - full suite result after the updates: `117 / 117` tests passed

**Recognition Loader Structural Split**:

- completed the first structural split of the monolithic recognition dataset loader while keeping the public loading API stable
- separated the internals into:
  - `recognition_dataset_splits.py` for dataset paths, mode resolution, logical split resolution, and deterministic sampling helpers
  - `recognition_dataset_protocol.py` for identity naming and the deterministic custom internal validation/probe split logic
  - `recognition_dataset_preprocessing.py` for sample-level loading, custom raw detection/preprocessing, pose collection, and quality filtering
  - `recognition_dataset_builders.py` for gallery/probe dataset builders
- kept `recognition_dataset_loader.py` as a thin compatibility facade so the public recognition loader API remains stable during the transition
- preserved the real attachment points that depend on the loader contract:
  - recognition trainer
  - recognition evaluator
  - system evaluator
  - existing loader tests that exercise split resolution and custom internal split semantics
- practical reading:
  - the recognition loader is now divided by responsibility in a way similar to the PAD loader split, while still respecting the extra protocol semantics that are specific to open-set recognition
  - future cleanup can now happen inside focused modules instead of revisiting one all-purpose loader file again

## 2026/05/27

**Neutral Runtime Naming Cleanup**:

- started the cleanup that turns the repository back into a neutral personal project instead of keeping the old exam-oriented runtime naming
- renamed the dedicated runtime entry points:
  - `src/main_cv.py` -> `src/main_pad.py`
  - `src/main_bs.py` -> `src/main_recognition.py`
- renamed the dedicated runtime pipeline modules:
  - `src/pipeline/run_pipeline_cv.py` -> `src/pipeline/run_pipeline_pad.py`
  - `src/pipeline/run_pipeline_bs.py` -> `src/pipeline/run_pipeline_recognition.py`
- updated the corresponding runtime function names and imports:
  - `run_pipeline_cv(...)` -> `run_pipeline_pad(...)`
  - `run_pipeline_bs()` -> `run_pipeline_recognition()`
- removed explicit `Computer Vision` / `Biometric Systems` wording from:
  - entry-point docstrings
  - runtime pipeline docstrings
  - runtime log prefixes
  - runtime stop/error messages
- the runtime variants are now described only by their function:
  - `full`
  - `pad-only`
  - `recognition-only`

**Helper Document Removal**:

- removed the old helper reports:
  - `recognition.md`
  - `antispoofing.md`
- decision rationale:
  - both files had become mostly historical experiment notes
  - both still carried the old `CV` / `BS` naming
  - their useful historical content remains recoverable through git history and the development log

**Test Alignment After Runtime Rename**:

- updated the runtime-helper tests to import the neutral module names
- renamed the affected helper-level test labels so they no longer refer to `cv` / `bs`
- kept the test logic unchanged: only module names and test naming were aligned

**PAD Documentation Pointer Cleanup**:

- removed the stale internal docstring reference from `pad_experiment_runner.py` that still pointed to `antispoofing.md`
- the PAD runner documentation now refers only to the active runtime/experiment configuration rather than to a removed side document

**Validation Status**:

- the runtime rename and helper-document removal were checked with:
  - full test suite
  - offline recognition experiment runner
  - offline system evaluator
  - runtime smoke checks for:
    - `main_pad`
    - `main_recognition`
    - `main`
- practical reading:
  - the cleanup changed naming and documentation surface, not the intended runtime or offline behavior
  - the project now has a cleaner base for the upcoming README rewrite

**Runtime CLI Helper Alignment And README Rewrite**:

- completed the first small CLI-alignment pass on the user-facing runtime entry points
- added minimal `argparse`-based `--help` support to:
  - `src/main.py`
  - `src/main_recognition.py`
- `src/main_pad.py` already exposed a runtime helper CLI, so the runtime entry points are now more uniform from a user perspective
- practical effect:
  - the runtime entry points can now be inspected safely with `--help` without opening the webcam loop
  - this makes the user-facing entry layer more consistent with the offline runners, which already exposed richer CLIs

- rewrote `README.md` from scratch as the main neutral project README
- removed the old exam-oriented wording and replaced it with project-level documentation covering:
  - project overview
  - main features
  - repository structure
  - installation
  - runtime commands
  - offline experiment commands
  - datasets
  - models and outputs
  - testing
  - command-line helper usage
- practical reading:
  - the repository now has a main README that matches the current codebase and naming
  - command discovery is clearer because the README explicitly documents which entry points support `--help`

2026/05/28

**Full Runtime PAD-Phase Overlay Clarification**:

- refined the third overlay line shown during the full runtime `PAD_CHECK` phase
- previously the lower status line stayed fixed on `Checking liveness...` even when the PAD overlay had already switched to `PAD: SPOOF`
- updated the text to reflect the visible PAD state more honestly:
  - `Checking liveness...` while PAD is still analyzing
  - `Recognition blocked` when PAD is showing spoof
  - `Recognition locked` when PAD is showing live before the runtime advances to the recognition-locked phase
- practical reading:
  - this does not change PAD or recognition logic
  - it only makes the user-facing overlay less misleading during liveness-only decisions
