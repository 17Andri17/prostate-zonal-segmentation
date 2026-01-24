# Prostate Zonal Segmentation

A comprehensive deep learning pipeline for segmenting prostate zones from multi-annotator MRI data using annotation fusion techniques.

## Project Overview

This project implements a U-Net based segmentation system for prostate zonal anatomy with support for handling multiple annotator labels. The system incorporates advanced annotation fusion methods including STAPLE (Simultaneous Truth and Performance Level Estimation) and custom multi-annotator fusion approaches to leverage multiple expert annotations.

### Key Features

- **Multi-Annotator Support**: Handle labels from multiple experts with reliability weighting
- **Annotation Fusion**: Implements STAPLE and custom fusion algorithms to combine multiple annotations
- **U-Net Architecture**: Fully convolutional segmentation network optimized for medical imaging
- **Mixed Precision Training**: GPU acceleration with automatic mixed precision
- **Probabilistic Labels**: Support for probabilistic fusion outputs as training targets
- **Comprehensive Evaluation**: Extensive metrics including Dice score, per-class evaluation, and visualization

## Project Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Trainer.py                         # Main training pipeline and evaluation
├── fusion_evaluation.py                # Fusion method comparison and evaluation
├── segmentation.ipynb                 # Complete segmentation workflow notebook
├── testing_notebook.ipynb             # Data loading and testing notebook
├── testDataUtils.py                   # Unit tests for data utilities
├── common_ids.txt                     # List of common patient IDs
│
├── Utils/
│   ├── UNet.py                        # U-Net architecture implementation
│   ├── DataUtils.py                   # Multi-annotator dataset classes
│   ├── AnnotationFusion.py            # Fusion algorithms (STAPLE, custom)
│   ├── Loss.py                        # Dice and Combined loss functions
│   └── utils.py                       # Helper utilities
│
└── exp/                               # Experiment results and models
    ├── FUSION_batch32_epochs100/
    ├── FUSION_100_epochs_DICE50_CE50/
    ├── STAPLE/
    ├── STAPLE_100_epochs/
    └── [other experimental variants]/
```

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Core Modules

### DataUtils.py

Implements multi-annotator dataset loading:

- **`MultiAnnotatorProstateDataset`**: Loads prostate MRI data with simulated multi-annotator annotations
  - Supports multiple modalities (T2W, coronal, sagittal)
  - Automatic normalization and resizing to target dimensions (256×256)
  - Simulated annotator variance to model realistic annotation disagreement
  - Loads anatomical labels and ProstateZones data

- **`MultiAnnotatorUNetDataset`**: Fusion-based dataset wrapper
  - Applies annotation fusion methods to generate ground truth
  - Supports both probabilistic and hard labels
  - Probabilistic labels for training, hard labels for validation/testing

### AnnotationFusion.py

Multi-annotator fusion algorithms:

- **`MultiAnnotatorFusion`**: Custom fusion approach
  - Pixel-level ambiguity computation
  - Annotator reliability estimation using weighted agreement
  - Probabilistic fusion combining reliability scores with class votes
  
- **`STAPLEFusionProvider`**: STAPLE algorithm implementation
  - Simultaneous Truth and Performance Level Estimation
  - EM-based optimization for performance parameters
  - Probabilistic label generation

### UNet.py

Segmentation architecture:

- **`UNet`**: Full U-Net implementation with configurable:
  - Number of input channels (default: 1 for single modality)
  - Number of output classes (default: 3 for prostate zones)
  - Bilinear or transposed convolution upsampling
  - Skip connections from encoder to decoder

### Loss.py

Training loss functions:

- **`DiceLoss`**: Soft Dice overlap metric as loss
  - Options to ignore background class
  - Smooth parameter to avoid division by zero

- **`CombinedLoss`**: Weighted combination of Dice and CrossEntropy
  - Configurable weights for each component
  - Class-weighted CrossEntropy to handle imbalanced classes

### Trainer.py

Complete training and evaluation pipeline:

- **`ProstateSegmentationTrainer`**: Main training orchestrator
  - Mixed precision training with GradScaler
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping with patience
  - Per-class metric tracking
  - Best model checkpointing
  - Comprehensive logging

Key methods:
- `train_epoch()`: Single training epoch with loss and Dice computation
- `validate()`: Validation with per-class metric collection
- `train()`: Full training loop with early stopping
- `evaluate()`: Test set evaluation with detailed metrics
- `plot_training_history()`: Visualization of training curves
- `visualize_predictions()`: Side-by-side ground truth vs prediction visualization

## Data Structure

The project expects data organized as follows:

```
Data/
├── <patient_id>/
│   ├── <patient_id>_t2w.mha          # T2-weighted MRI image
│   ├── <patient_id>_cor.mha          # Coronal view (optional)
│   └── <patient_id>_sag.mha          # Sagittal view (optional)

Anatomical_Labels/
├── <patient_id>/
│   └── <patient_id>_zones.mha        # Manual anatomical segmentation

ProstateZones/
├── <patient_id>/
│   └── Segmentation.mha              # ProstateX zones segmentation
```

## Segmentation Classes

The model segments prostate anatomy into 3 classes:

1. **NO-PG** (0): Background / Non-prostate
2. **PZ** (1): Peripheral Zone
3. **TZ** (2): Transition Zone

## Usage

### Quick Start - Training a Model

```python
python Trainer.py
```

This runs the default training setup with STAPLE fusion on your data.

### Custom Training Configuration

Edit `Trainer.py` at the bottom to customize:

```python
run_training(
    fusion_method=fusion_method,
    batch_size=4,              # Adjust batch size
    num_epochs=100,            # Number of training epochs
    num_workers=0,             # Set to number of CPU cores
    data_root="path/to/Data",
    labels_root="path/to/Labels",
    prostatex_root="path/to/Zones",
    experiment_name="exp/my_experiment"
)
```

### Evaluate Fusion Methods

```bash
python fusion_evaluation.py
```

Compares different annotation fusion approaches with statistical analysis.
