# Brain Tumor Detection and Segmentation Research Pipeline

A reproducible research prototype for brain tumor analysis from T1-weighted MRI using YOLO-based localization and SAM2-based segmentation.

The project is designed as an experimental framework rather than a clinical product. Its goal is to study whether detector-guided segmentation can provide a compact, reproducible pipeline for localizing and segmenting glioma, meningioma, and pituitary tumors.

## Research Question

Can a lightweight YOLO detector provide reliable spatial prompts for SAM2 and produce competitive tumor segmentation quality across tumor classes and MRI acquisition planes?

## Dataset

The primary dataset is **BRISC 2025**, published in *Scientific Data*.

BRISC contains:

- 6,000 contrast-enhanced T1-weighted MRI images
- 5,000 official training images
- 1,000 official test images
- Glioma, meningioma, pituitary tumor, and no-tumor classes
- Axial, coronal, and sagittal views
- Expert-reviewed pixel-level masks for tumor-positive scans
- Separate classification and segmentation task folders

Dataset:
https://www.kaggle.com/datasets/briscdataset/brisc2025

Paper:
https://doi.org/10.1038/s41597-026-06753-y

The dataset itself is intentionally not stored in this repository.

## Experimental Design

The repository treats the official BRISC test set as held-out evaluation data.

A validation split is created only from the official training partition. Stratification uses tumor class and MRI plane. Because public BRISC files do not provide patient identifiers, patient-disjoint splitting cannot be independently verified. This is recorded as a methodological limitation rather than hidden.

For detection, the target classes are:

| ID | Class |
|---:|---|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | Pituitary |

No-tumor scans are used as negative samples with empty YOLO label files instead of being represented as a bounding-box class.

Tumor bounding boxes are generated directly from the expert segmentation masks.

## Research Stages

### Stage 1 — Dataset Audit

The dataset is checked for:

- class distribution
- train/test counts
- MRI plane distribution
- segmentation-pair completeness
- empty masks
- tumor-mask area statistics

Run:

\`\`\`bash
python "Tumor detection.py" audit-brisc \
  --dataset-root /path/to/brisc2025 \
  --output outputs/dataset_audit
\`\`\`

### Stage 2 — Detection Dataset Preparation

Expert masks are converted into YOLO bounding boxes.

Run:

\`\`\`bash
python "Tumor detection.py" prepare-brisc \
  --dataset-root /path/to/brisc2025 \
  --output datasets/brisc_yolo \
  --validation-fraction 0.15 \
  --seed 42 \
  --clean
\`\`\`

Generated artifacts include:

- \`data.yaml\`
- train/validation/test image folders
- YOLO labels
- \`dataset_manifest.csv\`
- \`dataset_summary.json\`

### Stage 3 — Detector Training

A single model can be trained with:

\`\`\`bash
python "Tumor detection.py" train \
  --data datasets/brisc_yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 42
\`\`\`

### Stage 4 — Controlled Detector Experiments

The default study compares:

- YOLO11n
- YOLO11s
- YOLO11m
- 512 × 512 input
- 640 × 640 input

Run:

\`\`\`bash
python "Tumor detection.py" experiment \
  --data datasets/brisc_yolo/data.yaml \
  --models yolo11n.pt yolo11s.pt yolo11m.pt \
  --imgsz 512 640 \
  --epochs 100 \
  --batch 16 \
  --split val \
  --seed 42
\`\`\`

The script stores per-run results in CSV and JSON for direct comparison.

### Stage 5 — Detector Evaluation

Primary detection metrics are:

- Precision
- Recall
- mAP@50
- mAP@75
- mAP@50–95
- Inference time

Object detection is not evaluated with ordinary classification accuracy because localization quality matters in addition to class correctness.

Run:

\`\`\`bash
python "Tumor detection.py" evaluate \
  --weights runs/tumor_detection/<run>/weights/best.pt \
  --data datasets/brisc_yolo/data.yaml \
  --split val
\`\`\`

The official test split should be used only after model selection is complete.

### Stage 6 — YOLO-Guided SAM2 Segmentation

YOLO detections are converted into bounding-box prompts for SAM2.

Pipeline:

\`\`\`text
MRI
 ↓
YOLO tumor localization
 ↓
Bounding-box prompt
 ↓
SAM2
 ↓
Predicted tumor mask
 ↓
Comparison with expert BRISC mask
\`\`\`

Run:

\`\`\`bash
python "Tumor detection.py" benchmark-segmentation \
  --weights runs/tumor_detection/<run>/weights/best.pt \
  --dataset-root /path/to/brisc2025 \
  --sam-model sam2.1_b.pt \
  --split test \
  --imgsz 640
\`\`\`

Primary segmentation metrics are:

- Dice coefficient
- Intersection over Union
- Pixel precision
- Pixel recall
- Detection coverage

The benchmark also reports per-class segmentation performance.

## Metrics

### Detection

\`\`\`text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
mAP@50    = mean Average Precision at IoU 0.50
mAP@75    = mean Average Precision at IoU 0.75
mAP@50-95 = mean AP averaged across IoU 0.50 to 0.95
\`\`\`

### Segmentation

\`\`\`text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
\`\`\`

## Published BRISC Reference Results

The values below are **external published baselines**. They are included only as scientific context and are not claimed as results produced by this repository.

| Task | Model | Reported Result |
|---|---|---:|
| Classification | EfficientNetB0 | Accuracy 99.20% |
| Segmentation | Swin-HAFNet | Weighted mIoU 82.30% |
| Segmentation | U-Net | Weighted mIoU 75.70% |
| Segmentation | SaberNet | Weighted mIoU 80.60% |

Source: Fateh et al., *Scientific Data*, 2026.

These baselines provide reference points for interpreting future YOLO + SAM2 results. Direct comparison must account for differences in task formulation, training protocol, architecture, and evaluation setup.

## Result Integrity

This repository does not contain fabricated model scores.

Measured YOLO and SAM2 results are written automatically by the code after training and evaluation. Until a complete BRISC experiment is executed, published reference results remain clearly separated from repository-generated results.

This distinction is intentional because reproducibility and traceability are more important than displaying unverified accuracy numbers.

## Reproducibility

The pipeline records:

- random seed
- Python version
- PyTorch version
- Ultralytics version
- CUDA availability
- GPU name
- image size
- batch size
- training duration
- inference latency
- best checkpoint path
- experiment metrics

Experiment outputs are saved under \`runs/tumor_detection/\`.

## Repository Structure

\`\`\`text
Tumor-Detection/
├── Tumor detection.py
├── README.md
├── requirements.txt
└── .gitignore
\`\`\`

After dataset preparation and experimentation:

\`\`\`text
Tumor-Detection/
├── datasets/
│   └── brisc_yolo/
│       ├── images/
│       ├── labels/
│       ├── data.yaml
│       ├── dataset_manifest.csv
│       └── dataset_summary.json
├── runs/
│   └── tumor_detection/
├── Tumor detection.py
├── README.md
├── requirements.txt
└── .gitignore
\`\`\`

## Installation

Python 3.10 or newer is recommended.

\`\`\`bash
git clone https://github.com/Ibrahimshah0900/Tumor-Detection.git
cd Tumor-Detection
python -m venv .venv
\`\`\`

Windows:

\`\`\`bash
.venv\Scripts\activate
\`\`\`

Linux/macOS:

\`\`\`bash
source .venv/bin/activate
\`\`\`

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

For GPU training, install a PyTorch build appropriate for the local CUDA environment.

## Research Contributions of This Prototype

The current prototype focuses on four research-oriented ideas:

1. Converting expert tumor masks into reproducible detector supervision.
2. Comparing detector capacity and image resolution under a controlled protocol.
3. Using detection boxes as prompts for foundation-model segmentation.
4. Measuring the complete localization-to-segmentation pipeline against expert masks.

A natural next research extension is cross-plane generalization, uncertainty estimation, calibration, or comparison against conventional medical segmentation architectures such as U-Net and Attention U-Net.

## Limitations

- BRISC is composed of 2D MRI slices rather than full 3D volumes.
- Public patient identifiers are unavailable, so patient-disjoint splitting cannot be independently confirmed.
- The pipeline is a research prototype and is not intended for diagnosis or clinical decision-making.
- Detector-guided segmentation can fail when the detector misses the tumor.
- Results may vary with hardware, model version, initialization, and training configuration.

## Citation

Dataset paper:

\`\`\`bibtex
@article{fateh2026brisc,
  title={BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
  journal={Scientific Data},
  volume={13},
  article={361},
  year={2026},
  doi={10.1038/s41597-026-06753-y}
}
\`\`\`

## Author

**Muhammad Ibrahim Hashmi**  
BS Artificial Intelligence

Research interests: Computer Vision, Medical AI, Machine Learning, Deep Learning
