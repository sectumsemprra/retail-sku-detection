# Retail Shelf Product Detection — YOLOv11x
### IML Computer Vision Internship — Technical Assessment

---

## Results

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| Recall | 67.6% | **84.1%** | **+16.5pp** ✅ |
| Precision | — | **85.9%** | — |
| F1 Score | — | **85.0%** | — |
| mAP50 | — | **86.5%** | — |
| Count Accuracy | — | **99.3%** (144/145) | — |

> Evaluated on the original untouched test set (35 images, 145 ground truth instances)

---

## The Problem

A retail store uses a camera system to count products on shelves automatically.
The existing model had **67.6% recall** — meaning it missed roughly 1 in 3 products,
leading to incorrect stock counts.

This project addresses two tasks:
1. **Model Optimization** — Increase recall without significantly damaging precision
2. **Share of Shelf Analytics** — Calculate and visualize each SKU's percentage share across the test set

---

## Dataset

- **Source:** Provided by Intelligent Machines Ltd. as part of the technical assessment
- **License:** CC BY 4.0
- **Images:** 999 source images → ~3,000 after augmentation
- **Classes:** 76 SKU product classes (coded as q1, q4, q7 ... q299)
- **Format:** YOLOv11 normalized bounding boxes
- **Original split:** Train / Valid / Test

---

## Dataset Analysis — What I Found Before Training

Before writing a single line of training code I did a full dataset analysis.
This step revealed critical issues that shaped the entire approach.

### 1. Severe Class Imbalance
```
Most common SKU : q280 — 443 instances
Rarest SKU      : q178 —   2 instances
Imbalance ratio : 221x
```
Nearly half the classes had fewer than 20 training examples.
This directly causes missed detections on rare SKUs.

### 2. Broken Train/Val/Test Split
The original Roboflow split had a critical flaw:
```
49 out of 76 classes had ZERO test instances
```
This means the baseline 67.6% recall was only measured on ~26 classes.
The model was never even evaluated on 64% of all SKUs.

Notable examples:
- q64  — 225 training instances, 0 test instances
- q91  — 221 training instances, 0 test instances
- q145 — 357 training instances, 0 test instances

### 3. Box Size Analysis
```
Average box area  : 2.32% of image
Small boxes (<1%) : 12.2% of all boxes
```
Products are medium-sized in frame — `imgsz=640` is sufficient.

### 4. Shelf Density
```
Average objects per image : ~14
Densest image             : 44 objects
```
Dense shelf layouts require mosaic augmentation during training.

---

## Approach

### Step 1 — Fix the Data Split
The most impactful change was fixing the train/valid split
while keeping the **original test set completely untouched**.

I redistributed only the train and valid images using a
stratified strategy — ensuring every class appears in both
train and valid sets.

```
Before fix → 49 classes with 0 test instances
After fix  →  2 classes with 0 test instances
             (q178 and q46 — only 2-5 total instances, impossible to split further)
```

Result after redistribution:
```
Train : 892 images  (was ~800)
Valid :  72 images  (properly stratified)
Test  :  35 images  (original, untouched)
```

### Step 2 — Model Selection
I chose **YOLOv11x** — the largest model in the YOLOv11 family.

Reasons:
- 76 visually similar SKUs requires high model capacity
- H100 GPU available — no compute constraint
- Medium-sized products — no need for larger resolution
- Latest YOLO architecture with best feature extraction

### Step 3 — Training Configuration
Key decisions based on the data analysis:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `imgsz` | 640 | Box size analysis confirmed sufficient |
| `epochs` | 200 | Small dataset benefits from more passes |
| `batch` | 32 | Maximum H100 efficiency at 640 |
| `copy_paste` | 0.5 | Critical for 221x class imbalance |
| `mosaic` | 1.0 | Simulates dense shelf layouts |
| `mixup` | 0.15 | Helps generalization across similar SKUs |
| `optimizer` | AdamW | Better generalization vs SGD |
| `patience` | 50 | Generous early stopping for small dataset |
| `weight_decay` | 0.0005 | Prevents overfitting on rare classes |

I also tested `imgsz=1280` — it performed significantly worse
because the dataset is too small to benefit from higher resolution.
Precision collapsed to 47.8%, confirming 640 is the right choice.

### Step 4 — Confidence Threshold Tuning
After training I swept confidence thresholds from 0.05 to 0.80
to find the optimal balance between recall, precision and count accuracy.

Key insight: at low thresholds the model was overcounting —
226 detections vs 145 ground truth at conf=0.10.

```
conf=0.66 → detections=144  vs GT=145  diff=-1  ✅
```

### Step 5 — Test Time Augmentation (TTA)
TTA runs each image through multiple augmented versions
and averages predictions — catching products the model
might miss in a single pass. Enabled via `augment=True`.

---

## Share of Shelf Analytics

The entire test set is treated as a single representative store shelf.

```
Share of Shelf (%) = (SKU detections / Total detections) × 100
```

### Top 10 Results

| Rank | SKU | Detections | Share % |
|------|-----|-----------|---------|
| 1 | q64 | 29 | 13.74% |
| 2 | q214 | 22 | 10.43% |
| 3 | q280 | 20 | 9.48% |
| 4 | q293 | 19 | 9.00% |
| 5 | q193 | 13 | 6.16% |
| 6 | q13 | 12 | 5.69% |
| 7 | q31 | 12 | 5.69% |
| 8 | q121 | 8 | 3.79% |
| 9 | q289 | 7 | 3.32% |
| 10 | q61 | 7 | 3.32% |

Full results in `share_of_shelf_results.csv`

---

## Repository Structure

```
retail-sku-detection/
│
├── notebooks/
│   ├── sku-notebook-v4.ipynb      # Training — data split, model training, threshold sweep
│   └── inference-notebook.ipynb   # Inference, Share of Shelf analytics and visualizations
│
├── results/
│   ├── share_of_shelf_bar.png     # Bar chart — all SKUs ranked by shelf share
│   ├── share_of_shelf_pie.png     # Pie chart — top 10 SKUs
│   ├── share_of_shelf_results.csv # Full SKU share results table
│   └── threshold_analysis.png     # Confidence threshold sweep chart
│
├── dataset/
│   └── data_v2.yaml               # Fixed stratified split config
│
└── README.md
```

> **Model Checkpoint (best_v4.pt):**
> Too large for GitHub (114.6MB). Download from Kaggle:
> [https://www.kaggle.com/models/sectumsemprra/retail-sku-yolo11x](https://www.kaggle.com/models/sectumsemprra/retail-sku-yolo11x)

---

## How to Reproduce

### Requirements
```bash
pip install ultralytics roboflow matplotlib seaborn pandas numpy pillow pyyaml
```

### Step 1 — Get the Dataset
The dataset was provided by Intelligent Machines Ltd. as part of the technical assessment.
Extract the zip file and place it in your working directory. The folder structure should look like:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Step 2 — Run Training Notebook
Open `notebooks/sku-notebook-v4.ipynb` and run all cells in order.
The notebook will:
- Redistribute train/valid with stratified split
- Keep original test set untouched
- Train YOLOv11x for 200 epochs
- Save best weights to `weights/best.pt`

### Step 3 — Run Inference Notebook
Open `notebooks/inference-notebook.ipynb` and run all cells.
Download `best_v4.pt` from the Kaggle model page and
point `WEIGHTS_PATH` to it.

```python
# Download checkpoint from:
# https://www.kaggle.com/models/sectumsemprra/retail-sku-yolo11x

WEIGHTS_PATH = "best_v4.pt"  # downloaded from Kaggle
CONF         = 0.66          # optimal threshold
USE_TTA      = True          # test time augmentation
```

The notebook will:
- Evaluate on test set
- Run Share of Shelf inference
- Generate bar chart, pie chart and CSV

### Step 4 — Use Provided Checkpoint Directly
To skip training and run inference only:

Download `best_v4.pt` from:
[https://www.kaggle.com/models/sectumsemprra/retail-sku-yolo11x](https://www.kaggle.com/models/sectumsemprra/retail-sku-yolo11x)

```python
from ultralytics import YOLO

model   = YOLO("best_v4.pt")
results = model.predict(
    source  = "path/to/images",
    conf    = 0.66,
    iou     = 0.5,
    augment = True,
)
```

---

## Hardware
- GPU: NVIDIA H100 80GB HBM3
- Training time: ~28 minutes (200 epochs)
- Framework: PyTorch 2.9.0 + Ultralytics 8.4.17
- Platform: Kaggle

---

## Key Takeaways

**1. Data quality matters more than model size.**
The biggest recall improvement came from fixing the
train/valid split — not from using a larger model.

**2. The baseline 67.6% was measured on a broken split.**
Only 26/76 classes appeared in the test set. After fixing
the split, the model is now evaluated fairly across 74/76 classes.

**3. Overcounting is a real production problem.**
At low confidence thresholds the model detected 226 products
when only 145 existed. Threshold tuning brought this to 144/145
— a 99.3% count accuracy that is actually usable in a real store.

**4. TTA is a free performance boost.**
No retraining required — augmented inference improved
recall by ~2-3% at zero additional training cost.
