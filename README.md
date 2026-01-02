# LSTM_Mice
Long Short Term Memory Nueral Network for Recognizing Mice Social Behaviors 

# Mouse Behavior Detection with CNN-LSTM

A deep learning approach to classifying mouse social behaviors from pose keypoint tracking data.

## Overview

This project implements a hybrid CNN-LSTM neural network architecture with engineered features to detect and classify mouse behaviors from pose estimation data. The model processes sequences of anatomical keypoints (nose, ears, neck, hips, tail) tracked at 30Hz and predicts frame-level behavior labels.

### Key Features

- Hybrid CNN-LSTM architecture combining local pattern extraction with temporal sequence modeling
- Comprehensive feature engineering including inter-mouse distances, velocities, approach rates, and orientation angles
- Attention mechanism for interpretable predictions
- Focal loss with class weighting to handle severe class imbalance
- Mixup data augmentation for improved generalization

## Results

Model performance on CalMS21_task1 validation set:

| Behavior | F1 Score |
|----------|----------|
| intromit | 0.867 |
| mount | 0.856 |
| other | 0.809 |
| attack | 0.798 |
| sniff_any | 0.723 |
| genitalgroom | 0.547 |
| approach | 0.150 |
| **Macro Average** | **0.679** |



## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mabe-behavior-detection.git
cd mabe-behavior-detection

# Create conda environment
conda create -n mabe python=3.8
conda activate mabe

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn scipy
```

## Usage

### Data Preparation

Download the  data from [Kaggle](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) and place tracking and annotation files in the appropriate directories under `data/raw/`.

Process the data with feature engineering:

```bash
python scripts/load_calms21_engineered.py
```

This generates `data/processed/train_calms21_engineered.pkl` containing:
- Normalized engineered features (48 dimensions per frame)
- Frame-level behavior annotations
- Class vocabulary

  -- the code for this repo includes training off of one lab 

### Training

```bash
python scripts/train_cnn_lstm.py
```

Training configuration (modifiable in script):
- Window size: 60 frames
- Batch size: 64
- Learning rate: 0.0005
- Epochs: 100
- Early stopping patience: 20 epochs

The best model is automatically saved to `models/cnn_lstm_best.pth`.

### Inference

```python
import torch
from src.models.temporal_cnn_lstm import TemporalCNNLSTM

# Load model
checkpoint = torch.load('models/cnn_lstm_best.pth')
model = TemporalCNNLSTM(
    input_dim=48,
    num_classes=7,
    window_size=60
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    predictions = model(input_tensor)
    predicted_classes = predictions.argmax(dim=1)
```

## Model Architecture

### CNN-LSTM Network

```
Input: (batch_size, 60, 48) - 60 frames, 48 features per frame

CNN Feature Extractor:
├── Conv1D(48 -> 64, kernel=5) + BatchNorm + ReLU + MaxPool
├── Conv1D(64 -> 128, kernel=5) + BatchNorm + ReLU + MaxPool
└── Conv1D(128 -> 256, kernel=3) + BatchNorm + ReLU

Temporal Processor:
├── Bidirectional LSTM (256 -> 128, 2 layers)
└── Attention Mechanism

Classifier:
├── Linear(256 -> 128) + ReLU + Dropout
├── Linear(128 -> 64) + ReLU + Dropout
└── Linear(64 -> 7)

Output: (batch_size, 7) - class logits
```

### Engineered Features

The feature engineering module computes 48 features per frame:

| Category | Features | Description |
|----------|----------|-------------|
| Raw Keypoints | 28 | x,y coordinates for 7 bodyparts per mouse |
| Distance | 4 | Inter-mouse center, nose, and nose-to-body distances |
| Velocity | 6 | Per-mouse velocity (x, y) and speed |
| Acceleration | 2 | Per-mouse acceleration magnitude |
| Approach Rate | 2 | Rate of distance change between mice |
| Orientation | 4 | Per-mouse heading direction (sin, cos) |
| Facing Angle | 2 | Angle between heading and direction to other mouse |

## Training Details

### Loss Function

Focal Loss with class weighting addresses the severe class imbalance:

```
FL(p) = -alpha * (1 - p)^gamma * log(p)
```

Where `gamma=2.0` focuses learning on hard examples and `alpha` is inversely proportional to class frequency.

### Data Augmentation

- Gaussian noise injection (scale=0.02)
- Temporal jittering (shift by +/- 2 frames)
- Random scaling (0.95 to 1.05)
- Mixup augmentation (alpha=0.2)

### Class Distribution

The CalMS21_task1 subset exhibits significant class imbalance:

| Class | Samples | Percentage |
|-------|---------|------------|
| other | 63,542 | 68.0% |
| sniff_any | 15,234 | 16.3% |
| mount | 5,891 | 6.3% |
| attack | 4,567 | 4.9% |
| intromit | 2,134 | 2.3% |
| genitalgroom | 1,332 | 1.4% |
| approach | 708 | 0.8% |

## Future Work

- Train on full multi-lab dataset for improved generalization
- Implement graph neural networks for explicit relational modeling
- Explore transformer architectures for longer temporal dependencies
- Add ensemble methods combining multiple model architectures
- Develop post-processing temporal smoothing

## References

- [MABe Challenge Paper](https://arxiv.org/abs/2207.10553)
- [CalMS21 Dataset](https://sites.google.com/view/computational-behavior/our-datasets/calms21-dataset)



## Acknowledgments

- Multi-Agent Behavior Challenge organizers
- CalMS21 dataset contributors
- Kaggle competition platform
