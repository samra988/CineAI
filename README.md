# 🎬 Intelligent Movie Trailer Generation Pipeline

> An end-to-end AI system that automatically selects high-impact scenes, applies creepy visual transformations, generates NLP captions, and evaluates the final trailer — all from raw `.mp4` clips.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Design Justifications](#design-justifications)
- [Model Details](#model-details)
- [Limitations](#limitations)

---

## Overview

This project implements a fully automated movie trailer generation system. Given 18 short `.mp4` clips (each ~10 seconds), the pipeline:

1. Extracts rich visual features from every clip using deep learning
2. Trains a classifier to score each clip as **high-impact (+1)** or **low-impact (-1)**
3. Selects the **top 5 clips** and orders them for narrative flow
4. Applies **creepy horror-style visual effects** using OpenCV
5. Generates **NLP-based cinematic captions** using BLIP + rule-based rewriting
6. Burns captions directly onto frames and assembles the final trailers
7. Evaluates the trailer frame-by-frame and produces impact score graphs

---

## Pipeline Architecture

```
Raw Clips (0.mp4 - 17.mp4)
        |
        v
+-------------------------+
|  STEP 1: Feature        |
|  Extraction             |
|  - EfficientNet-B0      |  --> 1280-d CNN embedding per clip
|  - YOLO11-nano          |  --> 4-d object detection features
|  - Visual Dynamics      |  --> 5-d motion/brightness/cut features
|  = 1289-d feature vector|
+-------------------------+
        |
        v
+-------------------------+
|  STEP 2: Classification |
|  - Heuristic pseudo-    |
|    labels (no hardcode) |
|  - SVM (baseline)       |  --> Cross-validated, L2 regularized
|  - LSTM (temporal)      |  --> 2-layer, dropout=0.3, L2 Adam
|  - Ensemble (50/50)     |  --> Final impact score per clip
+-------------------------+
        |
        v
+-------------------------+
|  STEP 3: Trailer        |
|  Selection              |
|  - Top-5 by score       |
|  - Re-sorted by clip ID |  --> Chronological narrative order
+-------------------------+
        |
        v
+-------------------------+
|  STEP 4: Creepy         |
|  Transformation         |
|  - Desaturation/fog     |
|  - Darkening            |
|  - Red glowing eyes     |  --> YOLO-detected persons
|  - Glitch/scanlines     |
|  - Vignette overlay     |
+-------------------------+
        |
        v
+-------------------------+
|  STEP 5: NLP Captions   |
|  - BLIP image captioning|  --> Plain description per clip
|  - Rule-based rewrite   |  --> Sentiment modulation + keywords
|  - OpenCV text burn     |  --> Fade-in/out, dark bar, centred
+-------------------------+
        |
        v
+-------------------------+
|  STEP 6: Evaluation     |
|  - Frame-wise scoring   |
|  - Impact timeline graph|
|  - All-clips ranking    |
+-------------------------+
        |
        v
  trailer.mp4  +  creepy_trailer.mp4
```

---

## Features

### Step 1 - High-Impact Scene Identification
| Feature Group | Dimensions | Why it matters |
|---|---|---|
| EfficientNet-B0 CNN embeddings | 1280 | Captures rich visual semantics without manual engineering |
| YOLO11 object detection | 4 | Weapons, persons, unusual objects predict narrative tension |
| Motion & brightness dynamics | 5 | High motion = action; brightness spikes = explosions; cut score = editing pace |

### Step 2 - Classification Model
- **SVM (baseline):** RBF kernel, L2 regularization, 3-fold stratified cross-validation
- **LSTM (temporal):** 2-layer bidirectional LSTM, sliding window of 3 clips, dropout=0.3, L2 Adam weight decay
- **Ensemble:** 50/50 average of SVM and LSTM probabilities
- **No hardcoded labels:** Pseudo-labels derived from heuristic scores computed from the features themselves

### Step 3 - Trailer Selection
- Top-5 clips selected by ensemble score
- Re-ordered **chronologically** to preserve narrative flow: setup -> tension -> climax hint
- Diversity guaranteed: top-K from a ranked list naturally avoids redundant scenes

### Step 4 - Creepy Visual Transformation
Every frame of every selected clip is processed with:
- **Desaturation** (HSV saturation * 0.35) for a cold fog aesthetic
- **Darkening** (brightness * 0.60) for shadow depth
- **Glowing red eyes** on YOLO-detected persons (upper bounding box region)
- **Glitch/scanline** effect on every 5th frame (random pixel band shift)
- **Vignette** (Gaussian mask darkening corners)

### Step 5 - NLP Caption Generation
- **BLIP** (Salesforce) generates plain natural language descriptions
- **Rule-based rewriter** applies:
  - Verb replacement ("walks" -> "creeps", "stands" -> "looms")
  - Keyword injection (darkness, shadow, dread, haunted...)
  - Cinematic opener prefix ("They never saw what followed...")
- **OpenCV caption burning** (no ImageMagick / MoviePy TextClip required):
  - Word wrapping up to 3 lines
  - Semi-transparent dark bar for readability
  - White text with black stroke
  - Smooth fade-in (0.5 s) and fade-out

### Step 6 - Evaluation
- Frame-by-frame impact scoring on all 5 trailer clips
- Outputs a labelled timeline graph and per-clip bar chart
- Final trailer score = average across all evaluated frames
- Clips classified as +1 (high) or -1 (low) per the 0.35 threshold

---

## Project Structure

```
OEL/
|
|-- movie_trailer_pipeline.py   # Main pipeline (single runnable script)
|
|-- 0.mp4                       # Input clips
|-- 1.mp4
|-- ...
|-- 17.mp4
|
|-- output/                     # Auto-created on first run
    |-- trailer.mp4                 # Normal trailer (5 clips + captions)
    |-- creepy_trailer.mp4          # Horror-themed trailer
    |-- captioned_X.mp4             # Individual captioned clips
    |-- creepy_X.mp4                # Individual creepy clips
    |-- creepy_captioned_X.mp4      # Creepy + captions combined
    |-- impact_timeline.png         # Frame-wise score graph
    |-- all_clips_ranking.png       # All 18 clips ranked by score
    |-- captions.json               # Plain + creepy captions per clip
    |-- evaluation_results.json     # Per-clip evaluation stats
    |-- selection_metadata.json     # Why each clip was selected
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| CUDA (recommended) | 11.8+ |
| GPU VRAM | 4 GB minimum |
| Disk space | ~3 GB (models + output) |

### Tested on
- OS: Windows 10/11
- GPU: NVIDIA Quadro T2000 (4 GB VRAM)
- Python: 3.10

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-trailer-pipeline.git
cd movie-trailer-pipeline
```

### 2. Install dependencies

**PyTorch with CUDA (for NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**All other dependencies:**
```bash
pip install ultralytics opencv-python moviepy scikit-learn
pip install transformers Pillow matplotlib numpy tqdm
```

> **Note:** YOLO11 weights (`yolo11n.pt`) and BLIP weights download automatically on first run. Requires internet connection on first execution.

---

## Usage

1. Place your clips (`0.mp4`, `1.mp4`, ..., `17.mp4`) in the same folder as the script

2. Run the pipeline:
```bash
python movie_trailer_pipeline.py
```

3. All outputs are saved to the `output/` folder automatically

### Expected runtime (NVIDIA Quadro T2000)
| Step | Approximate Time |
|---|---|
| Feature extraction (18 clips) | 3-6 minutes |
| SVM + LSTM training | < 1 minute |
| Creepy transformation (5 clips) | 2-4 minutes |
| BLIP captioning | 1-2 minutes |
| Trailer assembly | 1-2 minutes |
| Evaluation + graphs | 1-2 minutes |
| **Total** | **~10-15 minutes** |

---

## Output Files

| File | Description |
|---|---|
| `trailer.mp4` | Final 50-second trailer (5 clips, captions overlaid) |
| `creepy_trailer.mp4` | Same trailer with full horror visual transformation |
| `impact_timeline.png` | Frame-wise impact score graph for each trailer clip |
| `all_clips_ranking.png` | Bar chart of all 18 clips ranked by ensemble score |
| `captions.json` | Plain (BLIP) and creepy (rewritten) captions per clip |
| `evaluation_results.json` | Per-clip avg/max score, high-impact %, final label |
| `selection_metadata.json` | Selected clip IDs, scores, and selection justification |

---

## Design Justifications

### Why pseudo-labels instead of manual annotation?
With only 18 clips and no ground-truth data, manually assigning labels would introduce human bias. Instead, pseudo-labels are derived from measurable visual signals (motion energy, object presence, brightness variance) -- a self-supervised approach that avoids hardcoding.

### Why SVM + LSTM ensemble?
- **SVM** excels at high-dimensional feature spaces (1289-d) and is robust with small datasets
- **LSTM** captures temporal ordering -- a clip that follows an exciting clip is more likely to be high-impact
- **Ensemble** reduces variance from either model's individual weaknesses

### Why k=3 cross-validation instead of k=5?
With 18 samples, k=5 folds would leave only 3-4 test samples per fold, making accuracy estimates unreliable. k=3 gives 6 test samples per fold for more stable evaluation.

### Why OpenCV for captions instead of MoviePy TextClip?
MoviePy's `TextClip` requires **ImageMagick** installed as a system binary. On most Windows machines this is missing or misconfigured, causing silent failures. `cv2.putText()` runs entirely in Python memory with zero external dependencies -- guaranteed to work on any machine.

### Why chronological ordering for the trailer?
Randomly ordering high-impact clips creates a jarring, incoherent experience. Sorting selected clips by their original index preserves the movie's narrative arc: early clips establish setup, middle clips build tension, later clips hint at the climax.

---

## Model Details

### EfficientNet-B0
- Pre-trained on ImageNet (1000 classes)
- Classifier head removed -- used as a 1280-d feature extractor
- No fine-tuning (frozen weights) -- suitable for small datasets

### YOLO11-nano
- Lightest YOLO11 variant (fits in 4 GB VRAM)
- Used for object detection only -- no training performed
- High-impact COCO classes manually defined (persons, weapons, vehicles, animals)

### BLIP (Salesforce/blip-image-captioning-base)
- Vision-Language model for image captioning
- Run in inference mode only (no fine-tuning)
- Applied to the middle frame of each selected clip

### LSTM Classifier
- 2-layer LSTM, hidden size 64, dropout 0.3
- Input: sliding window of 3 consecutive clip feature vectors
- Output: binary classification (high/low impact)
- Trained with Adam optimizer + L2 weight decay (1e-4) for regularization

---

## Limitations

- **No audio:** Pipeline is designed for silent clips. Audio features (MFCC, energy spikes) would further improve impact detection if audio is available.
- **Small dataset:** 18 clips is a very small training set. Pseudo-labels are a best-effort approximation of ground truth.
- **BLIP captions:** Quality depends on visual clarity of the sampled frame. Dark or blurry frames may produce generic captions.
- **Red eye effect:** Relies on YOLO detecting "person" class. If characters are too small, partially occluded, or non-human, the effect will not apply.
- **Fixed threshold (0.35):** The high/low impact threshold is a heuristic. Different movie genres may require tuning this value.

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) -- Object detection
- [Salesforce BLIP](https://github.com/salesforce/BLIP) -- Image captioning
- [EfficientNet (torchvision)](https://pytorch.org/vision/stable/models.html) -- CNN feature extraction
- [OpenCV](https://opencv.org/) -- Video processing and visual effects
- [scikit-learn](https://scikit-learn.org/) -- SVM classifier and cross-validation
- [PyTorch](https://pytorch.org/) -- LSTM model and deep learning infrastructure
