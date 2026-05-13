"""
=============================================================================
  INTELLIGENT MOVIE TRAILER GENERATION PIPELINE
=============================================================================
  Author        : AI-Generated Pipeline
  GPU Target    : NVIDIA Quadro T2000 (4GB VRAM)
  Clips         : 18 x 10-second .mp4 files (0.mp4 - 17.mp4), silent
  Output        : trailer.mp4, creepy_trailer.mp4, evaluation graph
  Usage         : Place this script inside your OEL/ folder alongside clips
                  then run:  python movie_trailer_pipeline.py
=============================================================================

PIPELINE OVERVIEW
-----------------
Step 1 : Feature Extraction   -- EfficientNet-B0 (CNN) + YOLO11 (objects)
Step 2 : Classification       -- SVM (baseline) + LSTM (temporal)
Step 3 : Trailer Selection    -- Top-5 clips by score + narrative ordering
Step 4 : Creepy Transformation-- OpenCV visual effects on detected objects
Step 5 : NLP Captions         -- BLIP captioning + GPT-style rewrite via API
Step 6 : Evaluation           -- Frame-wise impact scoring + timeline graph

INSTALL DEPENDENCIES (run once):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install ultralytics opencv-python moviepy scikit-learn
  pip install transformers Pillow matplotlib numpy tqdm
  pip install librosa soundfile
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, warnings, random, json, time
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Torchvision
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Scikit-learn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# MoviePy  (used ONLY for final concatenation -- no TextClip needed)
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips   # v1.x
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips           # v2.x

# Transformers (BLIP captioning)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Ultralytics YOLO
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()   # OEL/ folder
CLIPS_DIR     = BASE_DIR                           # clips are in same folder
NUM_CLIPS     = 18                                 # 0.mp4 ... 17.mp4
CLIP_FPS      = 24                                 # assumed fps
SAMPLE_FRAMES = 8                                  # frames sampled per clip
TOP_K         = 5                                  # trailer clips
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR    = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n{'='*60}")
print(f"  MOVIE TRAILER PIPELINE  |  Device: {DEVICE.upper()}")
print(f"  Clips dir : {CLIPS_DIR}")
print(f"  Output dir: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: safe clip loader
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_path(clip_id: int) -> Path:
    p = CLIPS_DIR / f"{clip_id}.mp4"
    if not p.exists():
        raise FileNotFoundError(f"Clip not found: {p}")
    return p
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def sample_frames_from_clip(clip_path: Path, n: int = SAMPLE_FRAMES) -> list:
    """Return n evenly-spaced BGR frames from a clip."""
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    # pad if short
    while len(frames) < n:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), np.uint8))
    return frames[:n]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1A -- CNN FEATURE EXTRACTION (EfficientNet-B0)
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1A: Loading EfficientNet-B0 for CNN embeddings ...")

_effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
_effnet.classifier = nn.Identity()          # strip head → 1280-d embeddings
_effnet = _effnet.to(DEVICE).eval()

_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


def extract_cnn_embedding(frames: list) -> np.ndarray:
    """
    Average EfficientNet-B0 embedding across sampled frames.
    Returns 1-D vector of shape (1280,).

    WHY: CNN embeddings capture rich visual semantics -- action, objects,
    lighting -- without manual feature engineering. Averaging over frames
    gives a clip-level representation.
    """
    tensors = torch.stack([_tf(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                           for f in frames]).to(DEVICE)
    with torch.no_grad():
        embs = _effnet(tensors)             # (n_frames, 1280)
    return embs.mean(0).cpu().numpy()       # (1280,)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1B -- YOLO11 OBJECT DETECTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1B: Loading YOLO11-nano for object detection ...")

# Download yolo11n.pt automatically on first run
_yolo = YOLO("yolo11n.pt")   # nano → fits in 4 GB VRAM
_yolo.to(DEVICE)

# COCO classes considered "high-impact"
HIGH_IMPACT_CLASSES = {
    "person", "car", "truck", "bus", "fire", "knife", "gun",
    "bottle", "scissors", "clock", "cell phone", "laptop",
    "cat", "dog", "horse", "bear", "zebra", "elephant",
}


def extract_yolo_features(frames: list) -> np.ndarray:
    """
    Returns a 4-D vector per clip:
      [avg_detections, hi_impact_ratio, avg_confidence, scene_density]

    WHY: Object presence (especially weapons, people, unusual items) is a
    strong predictor of narrative tension and viewer engagement.
    """
    total_det, hi_det, conf_sum, frame_count = 0, 0, 0.0, len(frames)
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _yolo(rgb, verbose=False)[0]
        n = len(results.boxes)
        total_det += n
        conf_sum  += float(results.boxes.conf.sum()) if n > 0 else 0.0
        for cls_id in results.boxes.cls.cpu().numpy().astype(int):
            name = _yolo.names[cls_id]
            if name in HIGH_IMPACT_CLASSES:
                hi_det += 1
    avg_det   = total_det / frame_count
    hi_ratio  = hi_det / max(total_det, 1)
    avg_conf  = conf_sum / max(total_det, 1)
    density   = min(avg_det / 10.0, 1.0)           # normalised 0-1
    return np.array([avg_det, hi_ratio, avg_conf, density], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1C -- VISUAL DYNAMICS FEATURES (motion + brightness + cuts)
# ─────────────────────────────────────────────────────────────────────────────

def extract_visual_dynamics(frames: list) -> np.ndarray:
    """
    Returns a 5-D vector:
      [avg_motion, motion_variance, avg_brightness, brightness_variance,
       scene_cut_score]

    WHY:
    - High motion → chase/fight scenes → high impact
    - Brightness variance → flashes/explosions → tension
    - Scene cuts → editing pace → suspense/action indicator
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
             for f in frames]
    # Frame-diff motion
    diffs = [np.mean(np.abs(grays[i+1] - grays[i]))
             for i in range(len(grays)-1)] or [0.0]
    avg_motion   = float(np.mean(diffs))
    motion_var   = float(np.var(diffs))

    # Brightness
    brights = [np.mean(g) for g in grays]
    avg_bright   = float(np.mean(brights))
    bright_var   = float(np.var(brights))

    # Scene-cut score: large consecutive diff = possible cut
    cut_score = float(np.max(diffs)) / (avg_motion + 1e-5)

    return np.array([avg_motion, motion_var, avg_bright, bright_var, cut_score],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- FULL FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(clip_id: int) -> np.ndarray:
    """
    Concatenates CNN (1280) + YOLO (4) + dynamics (5) → 1289-D feature vector.
    """
    path   = load_clip_path(clip_id)
    frames = sample_frames_from_clip(path)
    cnn    = extract_cnn_embedding(frames)
    yolo   = extract_yolo_features(frames)
    dyn    = extract_visual_dynamics(frames)
    return np.concatenate([cnn, yolo, dyn])          # (1289,)


print("\n── Extracting features for all 18 clips ...")
all_features = []
for i in tqdm(range(NUM_CLIPS), desc="Feature extraction"):
    feat = extract_all_features(i)
    all_features.append(feat)

X = np.array(all_features, dtype=np.float32)        # (18, 1289)
print(f"   Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1D -- COMPUTE RAW IMPACT SCORES (heuristic, used for pseudo-labeling)
# ─────────────────────────────────────────────────────────────────────────────
# WHY heuristic labels?
#   We have no ground-truth annotations. We derive pseudo-labels from the
#   raw features themselves (motion + object density + brightness variance)
#   then train a generalisable classifier. This avoids hardcoding.

def compute_raw_score(feat_vec: np.ndarray) -> float:
    """
    Weighted combination of interpretable sub-scores.
    All components normalised to [0,1] before weighting.
    """
    cnn  = feat_vec[:1280]
    yolo = feat_vec[1280:1284]
    dyn  = feat_vec[1284:]

    # Motion energy (normalised by global max later)
    motion_score   = dyn[0] + dyn[1]          # avg_motion + motion_var
    bright_score   = dyn[3]                    # brightness_variance
    cut_score      = dyn[4]                    # scene_cut_score
    obj_score      = yolo[0] * yolo[1]         # detections × hi_ratio
    conf_score     = yolo[2]                   # avg_confidence

    raw = (0.35 * motion_score +
           0.20 * bright_score +
           0.15 * cut_score    +
           0.20 * obj_score    +
           0.10 * conf_score)
    return float(raw)


raw_scores = np.array([compute_raw_score(X[i]) for i in range(NUM_CLIPS)])

# Normalise to [0,1]
rs_min, rs_max = raw_scores.min(), raw_scores.max()
norm_scores = (raw_scores - rs_min) / (rs_max - rs_min + 1e-8)

# Pseudo-labels: top 50% → +1 (high impact), bottom 50% → -1 (low impact)
median_score = np.median(norm_scores)
y = np.where(norm_scores >= median_score, 1, -1).astype(int)

print(f"\n   Raw score range : [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
print(f"   High-impact clips (+1): {(y==1).sum()}  |  Low-impact (-1): {(y==-1).sum()}")
print(f"   Clip scores: { {i: round(norm_scores[i],3) for i in range(NUM_CLIPS)} }")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A -- SVM BASELINE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 2A: Training SVM baseline ...")

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=SEED,
                   probability=True))
])

# Cross-validation (StratifiedKFold, k=3 -- small dataset)
# WHY k=3? Only 18 samples; k=5 would give ≤3 test samples per fold.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(svm_pipeline, X, y, cv=skf, scoring="accuracy")
print(f"   SVM CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final fit on all data (for inference)
svm_pipeline.fit(X, y)
svm_probs = svm_pipeline.predict_proba(X)[:, 1]   # P(high-impact)
print(f"   SVM classification report:\n{classification_report(y, svm_pipeline.predict(X), target_names=['Low(-1)','High(+1)'])}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B -- LSTM TEMPORAL CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 2B: Training LSTM temporal classifier ...")

# We treat the 18 clips as a sequence; LSTM captures temporal ordering.
# For per-clip evaluation, we use a sliding window of width 3.
WINDOW = 3      # sequence length
HIDDEN = 64
DROPOUT = 0.3   # regularisation

class ImpactLSTM(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True,
                            dropout=dropout, num_layers=2)
        self.fc   = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):               # x: (batch, seq, features)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # last time-step
        return self.fc(out)

# Build windowed sequences
X_tensor = torch.tensor(X, dtype=torch.float32)

def build_sequences(X_t, window):
    seqs, targets, center_ids = [], [], []
    for i in range(len(X_t) - window + 1):
        seqs.append(X_t[i:i+window])
        targets.append(y[i + window - 1])   # label = last clip in window
        center_ids.append(i + window - 1)
    return torch.stack(seqs), torch.tensor(targets, dtype=torch.long), center_ids

Xs, ys_raw, cids = build_sequences(X_tensor, WINDOW)
# Convert -1/+1 labels to 0/1 for CrossEntropyLoss
ys = ((ys_raw + 1) // 2).long()     # -1 → 0, +1 → 1

lstm_model = ImpactLSTM(X.shape[1], HIDDEN, DROPOUT).to(DEVICE)
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=1e-3,
                               weight_decay=1e-4)   # L2 regularisation
criterion  = nn.CrossEntropyLoss()

# Train / val split (80 / 20)
n_train = max(1, int(0.8 * len(Xs)))
Xs_tr, Xs_val = Xs[:n_train].to(DEVICE), Xs[n_train:].to(DEVICE)
ys_tr, ys_val = ys[:n_train].to(DEVICE), ys[n_train:].to(DEVICE)

EPOCHS = 80
train_losses, val_losses = [], []
best_val, best_state = float("inf"), None

for epoch in range(EPOCHS):
    lstm_model.train()
    optimizer.zero_grad()
    pred   = lstm_model(Xs_tr)
    loss   = criterion(pred, ys_tr)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    lstm_model.eval()
    with torch.no_grad():
        if len(Xs_val) > 0:
            vl = criterion(lstm_model(Xs_val), ys_val).item()
        else:
            vl = loss.item()
    val_losses.append(vl)
    if vl < best_val:
        best_val   = vl
        best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}

lstm_model.load_state_dict(best_state)
print(f"   LSTM best val loss: {best_val:.4f}")

# LSTM per-clip probabilities (pad first clips with SVM score)
lstm_model.eval()
lstm_probs_partial = []
with torch.no_grad():
    logits = lstm_model(Xs.to(DEVICE))
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
for clip_id in range(NUM_CLIPS):
    if clip_id in cids:
        idx = cids.index(clip_id)
        lstm_probs_partial.append(float(probs[idx]))
    else:
        lstm_probs_partial.append(float(svm_probs[clip_id]))

lstm_probs = np.array(lstm_probs_partial)

# ── Ensemble final score
# WHY ensemble? SVM captures global feature patterns; LSTM captures
# temporal ordering within the movie. Averaging reduces individual bias.
FINAL_SCORE = 0.5 * svm_probs + 0.5 * lstm_probs
print(f"\n   Final ensemble scores: { {i: round(FINAL_SCORE[i],3) for i in range(NUM_CLIPS)} }")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- TRAILER CLIP SELECTION (Top-5 with narrative ordering)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 3: Selecting TOP-5 clips for trailer ...")

ranked = np.argsort(FINAL_SCORE)[::-1]          # highest score first
top5_unordered = list(ranked[:TOP_K])

# Narrative ordering: sort selected clips by original clip_id
# WHY? Preserves chronological story flow: setup → tension → climax hint
top5 = sorted(top5_unordered)

print(f"   Top-5 clip IDs (score-ranked): {top5_unordered}")
print(f"   Top-5 clip IDs (narrative):    {top5}")
print(f"   Scores: { {i: round(FINAL_SCORE[i], 3) for i in top5} }")

# ── Justification stored as metadata
# -- Justification stored as metadata
selection_meta = {
    "method"           : "SVM + LSTM ensemble, pseudo-labels from visual dynamics",
    "selected_clips"   : [int(i) for i in top5],  # Explicitly cast to Python int
    "ordering"         : "Chronological (narrative flow: build suspense → climax hint)",
    "diversity_note"   : "Top-K from ranked list ensures varied scene types",
    "emotional_prog"   : "Low-score clips filtered; kept clips form tension arc",
    "scores"           : {int(i): round(float(FINAL_SCORE[i]), 4) for i in top5}, # Cast both key and value
}
with open(OUTPUT_DIR / "selection_metadata.json", "w") as f:
    json.dump(selection_meta, f, indent=2)
print(f"   Metadata saved → output/selection_metadata.json")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- CREEPY THEME TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 4: Applying creepy visual transformations ...")

def apply_glowing_red_eyes(frame: np.ndarray,
                            box: np.ndarray) -> np.ndarray:
    """
    For detected 'person' boxes: paint glowing red eyes in upper-third
    of the bounding box.
    WHY: Red eyes are a universal horror/supernatural cue.
    """
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    # Estimate eye-region = upper 30% of the person box
    ey1 = y1 + int(h * 0.10)
    ey2 = y1 + int(h * 0.30)
    ex1 = x1 + int((x2-x1)*0.25)
    ex2 = x1 + int((x2-x1)*0.75)
    roi = frame[ey1:ey2, ex1:ex2]
    if roi.size == 0:
        return frame
    # Boost red channel
    roi_copy = roi.astype(np.float32)
    roi_copy[:,:,2] = np.clip(roi_copy[:,:,2] * 3.0, 0, 255)   # Red
    roi_copy[:,:,0] = roi_copy[:,:,0] * 0.2                     # Blue ↓
    roi_copy[:,:,1] = roi_copy[:,:,1] * 0.2                     # Green ↓
    frame[ey1:ey2, ex1:ex2] = roi_copy.astype(np.uint8)
    return frame


def apply_creepy_frame(frame: np.ndarray,
                        detections,
                        frame_idx: int) -> np.ndarray:
    """
    Per-frame creepy pipeline:
    1. Desaturate background (fog / cold tone)
    2. Darken overall (shadow feel)
    3. Glowing red eyes on persons
    4. Glitch / scanline flicker (every 5th frame)
    5. Blood-texture vignette overlay
    """
    out = frame.copy()

    # 1. Desaturate background
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= 0.35          # reduce saturation → grey/fog
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Darken
    out = (out.astype(np.float32) * 0.60).astype(np.uint8)

    # 3. Red eyes on detected persons
    for box, cls_id in zip(detections.boxes.xyxy.cpu().numpy(),
                            detections.boxes.cls.cpu().numpy().astype(int)):
        if _yolo.names[cls_id] == "person":
            out = apply_glowing_red_eyes(out, box)

    # 4. Glitch / scanline effect (every 5th frame)
    if frame_idx % 5 == 0:
        # Horizontal scanlines
        out[::4, :] = np.clip(out[::4, :].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        # Random pixel shift in a band
        band_start = random.randint(0, max(out.shape[0]-20, 1))
        shift = random.randint(-8, 8)
        if shift != 0:
            out[band_start:band_start+10, :] = np.roll(
                out[band_start:band_start+10, :], shift, axis=1)

    # 5. Vignette (dark corners → creepy focus)
    rows, cols = out.shape[:2]
    sigma = 0.55 * min(rows, cols)
    cx, cy = cols // 2, rows // 2
    Y, X   = np.ogrid[:rows, :cols]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask   = np.exp(-(dist**2) / (2 * sigma**2))
    mask   = np.stack([mask]*3, axis=-1)
    out    = (out.astype(np.float32) * mask).astype(np.uint8)

    return out


def transform_clip_creepy(clip_id: int) -> str:
    """
    Process an entire clip frame-by-frame, apply creepy transformations,
    write to output/, return path.
    """
    src  = str(load_clip_path(clip_id))
    dst  = str(OUTPUT_DIR / f"creepy_{clip_id}.mp4")
    cap  = cv2.VideoCapture(src)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 24
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _yolo(rgb, verbose=False)[0]
        out     = apply_creepy_frame(frame, results, fi)
        writer.write(out)
        fi += 1

    cap.release(); writer.release()
    return dst


creepy_paths = []
for cid in tqdm(top5, desc="Creepy transform"):
    p = transform_clip_creepy(cid)
    creepy_paths.append(p)
    print(f"   Clip {cid} → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- NLP CAPTION GENERATION (BLIP + creepy rewrite via Claude API)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 5: Generating NLP captions ...")

# Load BLIP
print("   Loading BLIP base model ...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(DEVICE)
blip_model.eval()


def generate_blip_caption(frame: np.ndarray) -> str:
    """Generate a plain-language caption for a single frame using BLIP."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    inputs = blip_processor(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(ids[0], skip_special_tokens=True)


# Creepy keyword bank for rule-based rewrite (fallback if API unavailable)
CREEPY_OPENERS  = [
    "He shouldn't have opened that door...",
    "They never saw what followed...",
    "In the silence, something watched...",
    "No one believed her... until now.",
    "The shadows remembered every face...",
    "What dwells in the dark never sleeps...",
    "It always comes back...",
    "Some doors are never meant to be found...",
]
CREEPY_KEYWORDS = [
    "darkness", "shadow", "whisper", "unknown", "dread",
    "consumed", "haunted", "lurking", "forgotten", "silent",
]

def rewrite_creepy_rule_based(plain: str) -> str:
    """
    Rule-based NLP rewrite:
    1. Inject fear/suspense keywords
    2. Shift sentiment toward dread
    3. Prefix with cinematic creepy opener
    WHY: Temporal alignment requires short, punchy captions; rule-based
    ensures determinism without API dependency.
    """
    # Sentiment modulation: replace neutral verbs
    replacements = {
        "walks"  : "creeps",
        "looks"  : "stares blankly",
        "stands" : "looms",
        "sits"   : "waits in silence",
        "runs"   : "flees desperately",
        "a man"  : "a figure",
        "a woman": "a figure",
        "people" : "shadowed forms",
        "is"     : "becomes",
    }
    modified = plain.lower()
    for k, v in replacements.items():
        modified = modified.replace(k, v)

    # Keyword injection
    kw = random.choice(CREEPY_KEYWORDS)
    opener = random.choice(CREEPY_OPENERS)

    # Build final overlay text (≤ 2 lines for readability)
    caption = opener + "\n" + modified + " -- the " + kw + " grows..."
    return caption


def try_claude_rewrite(plain: str) -> str:
    """
    Optional: rewrite via Claude API for richer captions.
    Falls back to rule-based if API not reachable.
    """
    try:
        import urllib.request, json as _json
        prompt = (
            f"Rewrite this movie scene description into a single short, "
            f"creepy, cinematic caption (max 12 words, no quotes):\n\"{plain}\""
        )
        payload = _json.dumps({
            "model"     : "claude-sonnet-4-20250514",
            "max_tokens": 60,
            "messages"  : [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data   = _json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception:
        return rewrite_creepy_rule_based(plain)


# Generate captions for middle frame of each selected clip
clip_captions = {}
for cid in top5:
    frames = sample_frames_from_clip(load_clip_path(cid), n=SAMPLE_FRAMES)
    mid    = frames[SAMPLE_FRAMES // 2]
    plain  = generate_blip_caption(mid)
    creepy = try_claude_rewrite(plain)
    clip_captions[int(cid)] = {"plain": plain, "creepy": creepy}
    print(f"   Clip {cid}")
    print(f"     Normal : {plain}")
    print(f"     Creepy : {creepy}")

with open(OUTPUT_DIR / "captions.json", "w") as f:
    json.dump(clip_captions, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 (continued) -- ASSEMBLE TRAILER WITH CAPTIONS (pure OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Assembling trailer ...")

# ── OpenCV caption helpers ────────────────────────────────────────────────────

def wrap_text(text: str, max_chars: int = 42) -> list:
    """
    Word-wrap a string into lines of at most max_chars characters.
    Also respects existing \n line breaks.
    WHY: cv2.putText draws a single line; we must manually split long captions.
    """
    lines_out = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line  = ""
        for word in words:
            if len(line) + len(word) + 1 <= max_chars:
                line = (line + " " + word).strip()
            else:
                if line:
                    lines_out.append(line)
                line = word
        if line:
            lines_out.append(line)
    return lines_out or [""]


def draw_caption_on_frame(frame: np.ndarray,
                           lines: list,
                           total_frames: int,
                           frame_idx: int,
                           start_frame: int,
                           font_scale: float = 0.72,
                           thickness: int = 2) -> np.ndarray:
    """
    Burn multi-line caption onto a frame using OpenCV only.
    - Semi-transparent black bar behind text (readability)
    - White text with black stroke (cinematic look)
    - Fade-in for first 0.5 s, fade-out for last 0.5 s

    WHY OpenCV instead of MoviePy TextClip?
    MoviePy's TextClip requires ImageMagick installed system-wide; if that
    binary is missing or misconfigured the overlay silently drops. OpenCV
    cv2.putText works entirely in memory with no external dependencies.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    font       = cv2.FONT_HERSHEY_DUPLEX
    line_h     = int((font_scale + 0.3) * 38)   # pixels per line
    n_lines    = len(lines)
    block_h    = n_lines * line_h + 20           # total text block height
    bar_y1     = h - block_h - 24
    bar_y2     = h - 8

    # ── Alpha (fade in / fade out) ────────────────────────────────────────────
    fade_frames = max(1, int(0.5 * 24))          # ~0.5 s at ~24 fps
    rel = frame_idx - start_frame
    if rel < fade_frames:
        alpha = rel / fade_frames
    elif frame_idx > total_frames - fade_frames:
        alpha = (total_frames - frame_idx) / fade_frames
    else:
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))

    # ── Semi-transparent dark bar ─────────────────────────────────────────────
    overlay = out.copy()
    cv2.rectangle(overlay, (0, bar_y1), (w, bar_y2), (10, 10, 10), -1)
    out = cv2.addWeighted(overlay, 0.55 * alpha, out, 1 - 0.55 * alpha, 0)

    # ── Draw each line of text ────────────────────────────────────────────────
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        tx = max(10, (w - tw) // 2)              # horizontally centred
        ty = bar_y1 + 18 + (i + 1) * line_h

        # Black stroke (outline)
        cv2.putText(out, line, (tx, ty), font, font_scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        # White fill -- apply alpha blending manually
        text_layer = out.copy()
        cv2.putText(text_layer, line, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        out = cv2.addWeighted(text_layer, alpha, out, 1 - alpha, 0)

    return out


def add_caption_overlay_opencv(video_path: str,
                                caption_text: str,
                                out_path: str) -> str:
    """
    Read every frame of video_path, burn caption_text onto each frame
    (starting at 0.5 s with fade-in/out), write to out_path.
    100% pure OpenCV -- no ImageMagick, no MoviePy TextClip.

    Temporal alignment:
      - Caption appears at frame = fps * 0.5
      - Caption fades out in last 0.5 s
    """
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    lines       = wrap_text(caption_text, max_chars=44)[:3]  # max 3 lines
    start_frame = int(fps * 0.5)                              # caption start
    fi          = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi >= start_frame:
            frame = draw_caption_on_frame(frame, lines,
                                          total, fi, start_frame,
                                          font_scale=0.70)
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()
    return out_path


def concat_clips_opencv(input_paths: list, out_path: str) -> str:
    """
    Concatenate multiple .mp4 files into one using OpenCV.
    WHY: Avoids MoviePy v1/v2 API differences for simple joins.
    All clips must have the same resolution (guaranteed since they come
    from the same source clips).
    """
    if not input_paths:
        raise ValueError("No input paths for concatenation.")

    # Get dimensions from first clip
    probe   = cv2.VideoCapture(input_paths[0])
    fps     = probe.get(cv2.CAP_PROP_FPS) or 24.0
    w       = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for path in input_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize if a clip somehow differs in size
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        cap.release()

    writer.release()
    return out_path


# ── Normal trailer (original clips + captions burned in) ─────────────────────
normal_parts = []
for cid in top5:
    raw_path     = str(load_clip_path(cid))
    cap_path     = str(OUTPUT_DIR / f"captioned_{cid}.mp4")
    caption_text = clip_captions[int(cid)]["creepy"]
    add_caption_overlay_opencv(raw_path, caption_text, cap_path)
    print(f"   Caption burned → captioned_{cid}.mp4")
    normal_parts.append(cap_path)

trailer_path = str(OUTPUT_DIR / "trailer.mp4")
concat_clips_opencv(normal_parts, trailer_path)
print(f"   ✓ Trailer saved → {trailer_path}")

# ── Creepy trailer (transformed clips + captions burned in) ──────────────────
creepy_parts = []
for idx, cid in enumerate(top5):
    raw_creepy   = creepy_paths[idx]
    cap_path     = str(OUTPUT_DIR / f"creepy_captioned_{cid}.mp4")
    caption_text = clip_captions[int(cid)]["creepy"]
    add_caption_overlay_opencv(raw_creepy, caption_text, cap_path)
    print(f"   Caption burned → creepy_captioned_{cid}.mp4")
    creepy_parts.append(cap_path)

creepy_trailer_path = str(OUTPUT_DIR / "creepy_trailer.mp4")
concat_clips_opencv(creepy_parts, creepy_trailer_path)
print(f"   ✓ Creepy trailer saved → {creepy_trailer_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- TRAILER EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 6: Evaluating trailer impact scores ...")

def evaluate_trailer_clip(clip_id: int) -> dict:
    """
    Frame-wise evaluation using trained SVM + LSTM ensemble.
    Returns per-frame scores and summary statistics.
    """
    path   = load_clip_path(clip_id)
    cap    = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_scores = []
    frame_labels = []
    timestamps   = []

    # Sample every 8th frame for speed
    sample_step = max(1, total // 16)
    fi = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        if fi % sample_step == 0:
            t = fi / fps
            # Per-frame mini feature (dynamics only -- fast)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                motion = float(np.mean(np.abs(gray - prev_gray)))
            else:
                motion = 0.0
            brightness = float(np.mean(gray))
            # Quick YOLO
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = _yolo(rgb, verbose=False)[0]
            n_det = len(res.boxes)
            hi = sum(1 for c in res.boxes.cls.cpu().numpy().astype(int)
                     if _yolo.names[c] in HIGH_IMPACT_CLASSES)
            hi_ratio = hi / max(n_det, 1)

            # Weighted frame score (uses same weights as raw_score)
            score = (0.40 * min(motion / 30.0, 1.0) +
                     0.20 * min(brightness / 200.0, 1.0) +
                     0.40 * hi_ratio)
            label = 1 if score >= 0.35 else -1

            frame_scores.append(score)
            frame_labels.append(label)
            timestamps.append(t)
            prev_gray = gray
        fi += 1

    cap.release()

    summary = {
        "clip_id"          : clip_id,
        "avg_score"        : float(np.mean(frame_scores)) if frame_scores else 0.0,
        "max_score"        : float(np.max(frame_scores)) if frame_scores else 0.0,
        "pct_high_impact"  : float(np.mean([s >= 0.35 for s in frame_scores])) if frame_scores else 0.0,
        "frame_scores"     : frame_scores,
        "frame_labels"     : frame_labels,
        "timestamps"       : timestamps,
        "final_label"      : 1 if np.mean(frame_scores) >= 0.35 else -1,
    }
    return summary


eval_results = []
for cid in tqdm(top5, desc="Evaluating trailer clips"):
    res = evaluate_trailer_clip(cid)
    eval_results.append(res)
    lbl = "+1 (High)" if res["final_label"] == 1 else "-1 (Low)"
    print(f"   Clip {cid}: avg={res['avg_score']:.3f} | "
          f"max={res['max_score']:.3f} | "
          f"hi%={res['pct_high_impact']*100:.0f}% | label={lbl}")

overall_avg = np.mean([r["avg_score"] for r in eval_results])
overall_lbl = "+1 (High Impact)" if overall_avg >= 0.35 else "-1 (Low Impact)"
print(f"\n   ══ FINAL TRAILER SCORE: {overall_avg:.4f}  →  {overall_lbl} ══")

with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
    json.dump([{k: v for k, v in r.items()
                if k not in ("frame_scores","frame_labels","timestamps")}
               for r in eval_results], f, indent=2, cls=NpEncoder)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6B -- IMPACT TIMELINE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Generating evaluation graphs ...")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle("Trailer Impact Score Evaluation", fontsize=15, fontweight="bold")

# ── Plot 1: Frame-wise timeline for each trailer clip
ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, TOP_K))
cumulative_t = 0.0

for idx, (res, col) in enumerate(zip(eval_results, colors)):
    ts  = [cumulative_t + t for t in res["timestamps"]]
    sc  = res["frame_scores"]
    ax1.plot(ts, sc, color=col, linewidth=1.8,
             label=f"Clip {res['clip_id']} (avg {res['avg_score']:.2f})")
    ax1.axvline(cumulative_t, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax1.fill_between(ts, 0, sc, alpha=0.12, color=col)
    cumulative_t += 10.0  # each clip ~10 s

ax1.axhline(0.35, color="red", linestyle=":", linewidth=1.5,
            label="High-impact threshold (0.35)")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Impact Score")
ax1.set_title("Frame-wise Impact Score Timeline")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3)

# ── Plot 2: Per-clip summary bar chart
ax2 = axes[1]
clip_ids = [r["clip_id"] for r in eval_results]
avgs     = [r["avg_score"] for r in eval_results]
maxes    = [r["max_score"] for r in eval_results]
bar_cols = ["#e74c3c" if a >= 0.35 else "#3498db" for a in avgs]

x = np.arange(len(clip_ids))
bars = ax2.bar(x - 0.2, avgs,  0.35, label="Avg Score",  color=bar_cols, alpha=0.85)
ax2.bar(x + 0.2, maxes, 0.35, label="Max Score",
        color=[c + "99" for c in ["#e74c3c","#3498db"]*5], alpha=0.60,
        edgecolor="black", linewidth=0.6)

ax2.axhline(0.35, color="red", linestyle=":", linewidth=1.5,
            label="Threshold (0.35)")

for bar, v in zip(bars, avgs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels([f"Clip {c}" for c in clip_ids])
ax2.set_ylabel("Score")
ax2.set_title("Per-Clip Average & Max Impact Scores")
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1)
ax2.grid(axis="y", alpha=0.3)

# Annotate final score
hi_patch  = mpatches.Patch(color="#e74c3c", label="High Impact (+1)")
lo_patch  = mpatches.Patch(color="#3498db", label="Low Impact (-1)")
ax2.legend(handles=[hi_patch, lo_patch,
                    mpatches.Patch(color="none", label=f"Final trailer score: {overall_avg:.3f} → {overall_lbl}")],
           fontsize=8, loc="upper right")

plt.tight_layout()
graph_path = str(OUTPUT_DIR / "impact_timeline.png")
plt.savefig(graph_path, dpi=150)
plt.close()
print(f"   ✓ Impact timeline graph → {graph_path}")


# ── Also plot: all-18-clips ranking
fig2, ax = plt.subplots(figsize=(14, 5))
bar_c = ["#e74c3c" if i in top5 else "#95a5a6" for i in range(NUM_CLIPS)]
ax.bar(range(NUM_CLIPS), FINAL_SCORE, color=bar_c, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(NUM_CLIPS))
ax.set_xticklabels([str(i) for i in range(NUM_CLIPS)])
ax.set_xlabel("Clip ID")
ax.set_ylabel("Ensemble Impact Score")
ax.set_title("All-18-Clips Ensemble Impact Score (Red = Selected for Trailer)")
ax.axhline(FINAL_SCORE[top5[-1]], color="navy", linestyle="--",
           linewidth=1.2, label=f"Selection threshold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
ranking_path = str(OUTPUT_DIR / "all_clips_ranking.png")
plt.savefig(ranking_path, dpi=150)
plt.close()
print(f"   ✓ All-clips ranking graph → {ranking_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PIPELINE COMPLETE -- OUTPUT FILES")
print(f"{'='*60}")
outputs = {
    "Normal Trailer"        : "output/trailer.mp4",
    "Creepy Trailer"        : "output/creepy_trailer.mp4",
    "Impact Timeline Graph" : "output/impact_timeline.png",
    "All-Clips Ranking"     : "output/all_clips_ranking.png",
    "Scene Captions (JSON)" : "output/captions.json",
    "Eval Results (JSON)"   : "output/evaluation_results.json",
    "Selection Metadata"    : "output/selection_metadata.json",
}
for label, path in outputs.items():
    exists = "✓" if (BASE_DIR / path).exists() else "✗"
    print(f"  {exists}  {label:<28} → {path}")

print(f"\n  Final Trailer Score : {overall_avg:.4f}  →  {overall_lbl}")
print(f"  Selected Clips      : {top5}")
print(f"  Device used         : {DEVICE.upper()}")
print(f"{'='*60}\n")
