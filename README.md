## Thesis Translation Utilities

This project contains small utilities to translate the PHOENIX-2014-T dataset from German to English.

### Prerequisites
- Python 3.9+
- Google Cloud project with the Cloud Translation API enabled
- A service account key JSON file named `credentials.json`

### Installation
```bash
pip install pandas google-cloud-translate
```

Place `credentials.json` in the project root or set the env var to its full path:
```bash
set GOOGLE_APPLICATION_CREDENTIALS=credentials.json
```

### Files
- `translation.py`: Translates the `orth` and `translation` columns to English (`orth_english`, `translation_english`).

### Usage
1) Translate source CSV (edit input path inside the script as needed):
```bash
python translation.py
```
Output: the translated version of the input file (Excel‑friendly: UTF‑8 BOM and all fields quoted).



### Excel/CSV Notes
- Because translations often contain commas, outputs are saved with all fields quoted and UTF‑8 BOM so Excel imports the full text into single cells.
- If opening in Excel via Data → From Text/CSV, confirm delimiter is comma and encoding is UTF‑8.

# Sign Language Transformers: Joint End-to-End Recognition and Translation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Replication and English adaptation of "Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation" (Camgöz et al., CVPR 2020)

This project implements a joint sign language recognition and translation system using transformer networks, adapted from German to English Sign Language on the PHOENIX-2014-T dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

This system performs two tasks simultaneously:

1. **Sign Language Recognition (CSLR)**: Recognizes sign glosses from continuous sign language videos
2. **Sign Language Translation (SLT)**: Translates sign language videos directly to English sentences

**Key Innovation**: Joint training with shared encoder improves both tasks through multi-task learning.

### Architecture

```
Video Frames → CNN → Transformer Encoder ─┬→ CTC Loss (Recognition)
                                          └→ Transformer Decoder → CrossEntropy Loss (Translation)
```

## ✨ Features

- ✅ Complete end-to-end training pipeline
- ✅ Joint recognition and translation with shared encoder
- ✅ CTC loss for alignment-free recognition
- ✅ Transformer-based architecture (3 layers, 8 heads, 512 hidden)
- ✅ TensorBoard integration for real-time monitoring
- ✅ Automatic checkpointing and best model saving
- ✅ Comprehensive visualization and logging
- ✅ Configurable hyperparameters via YAML

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- 100GB+ disk space for dataset

### Setup

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed experiments/runs
```

### Dependencies

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
imageio>=2.31.0
Pillow>=10.0.0
pyyaml>=6.0
tensorboard>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## 📊 Dataset Setup

### 1. Download PHOENIX-2014-T Dataset

Download the RWTH-PHOENIX-Weather 2014 T dataset from:
- [Official Dataset Page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

Expected structure:
```
PHOENIX-2014-T/
├── features/
│   └── fullFrame-210x260px/
│       ├── train/
│       ├── dev/
│       └── test/
└── annotations/
    └── manual/
        ├── PHOENIX-2014-T.train.corpus.csv
        ├── PHOENIX-2014-T.dev.corpus.csv
        └── PHOENIX-2014-T.test.corpus.csv
```

### 2. Prepare English Translations

Translate German annotations to English or use provided English translations:
```
data/processed/annotations/
├── PHOENIX-2014-T.train.corpus_eng.csv
├── PHOENIX-2014-T.dev.corpus_eng.csv
└── PHOENIX-2014-T.test.corpus_eng.csv
```

Each CSV should contain:
- `name`: Video identifier
- `orth_english`: English glosses (e.g., "NOW WEATHER TOMORROW")
- `translation_english`: English sentence (e.g., "and now the weather forecast")

### 3. Update Configuration

Edit `config.yaml` with your dataset paths:
```yaml
data:
  train_annotations: /path/to/PHOENIX-2014-T.train.corpus_eng.csv
  train_features: /path/to/features/fullFrame-210x260px/train
  # ... (dev and test paths)
```

## 🚀 Quick Start

### Train the Model

```bash
# Start training with default configuration
python scripts/05_train.py

# Monitor with TensorBoard
tensorboard --logdir=experiments/runs/exp_YYYYMMDD_HHMMSS/logs
# Open browser to http://localhost:6006
```

### Training Output

```
================================================================================
Starting Training
================================================================================
Epochs: 100
Device: cuda
Experiment directory: experiments/runs/exp_20251003_092824
Training samples: 7096
Validation samples: 519
Batch size: 4
Lambda Recognition: 5.0
Lambda Translation: 1.0
================================================================================

Epoch 1 [Train]:   0%|          | 0/1774 [00:00<?, ?it/s]
```

## 📁 Project Structure

```
sign-language-transformers/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   └── vocabulary.py      # Vocabulary building
│   ├── models/
│   │   ├── positional_encoding.py
│   │   ├── slrt.py           # Recognition Transformer
│   │   ├── sltt.py           # Translation Transformer
│   │   └── joint_model.py    # Combined model
│   ├── training/
│   │   ├── losses.py         # Joint loss function
│   │   └── trainer.py        # Training loop
│   └── utils/
│       └── logging.py        # TensorBoard + plotting
│
├── scripts/
│   └── 05_train.py           # Main training script
│
├── experiments/
│   └── runs/
│       └── exp_YYYYMMDD_HHMMSS/
│           ├── config.json
│           ├── checkpoints/
│           │   ├── best_model.pth
│           │   └── checkpoint_epoch_*.pth
│           └── logs/
│               ├── events.out.tfevents.*
│               └── plots/
│                   ├── training_curves.png
│                   └── train_val_comparison.png
│
└── data/
    ├── raw/                  # Original dataset
    └── processed/            # Processed annotations
```

## ⚙️ Configuration

Key parameters in `config.yaml`:

### Model Architecture
```yaml
model:
  input_dim: 1024        # CNN output dimension
  hidden_dim: 512        # Transformer hidden size
  num_layers: 3          # Number of transformer layers
  num_heads: 8           # Number of attention heads
  dropout: 0.1           # Dropout rate
```

### Training Hyperparameters
```yaml
training:
  batch_size: 4                # increase batch size 128->64->48
  learning_rate: 0.001         # Initial learning rate
  num_epochs: 100              # Total epochs
  lambda_recognition: 5.0      # Recognition loss weight
  lambda_translation: 1.0      # Translation loss weight
  gradient_clip: 1.0           # Gradient clipping threshold
  device: cuda                 # 'cuda', 'mps', or 'cpu'
  num_workers: 0               # DataLoader workers (0 for Windows)
```

## 🎓 Training

### Multi-Task Loss Function

The model uses a weighted combination of two losses:

**L = λR × CTC_Loss + λT × CrossEntropy_Loss**

- **CTC Loss** (λR = 5.0): For recognition (handles variable-length alignment)
- **Cross-Entropy Loss** (λT = 1.0): For translation (word-by-word prediction)

### Training Process

1. **Data Loading**: Batches of videos, glosses, and sentences
2. **Forward Pass**: Through CNN → Encoder → Decoder
3. **Loss Calculation**: Combined recognition + translation loss
4. **Backpropagation**: Compute gradients
5. **Optimization**: Update model parameters
6. **Validation**: Evaluate on dev set
7. **Checkpointing**: Save best model

### Learning Rate Schedule

- Starts at 0.001
- Reduces by 0.7× when validation loss plateaus (patience=8)
- Stops when LR < 1e-6

## 📈 Evaluation

### Metrics

- **Recognition**: Word Error Rate (WER) - lower is better
- **Translation**: BLEU-4 Score - higher is better

### Expected Performance (from paper)

| Task | Metric | Target |
|------|--------|--------|
| Recognition | WER | ~24-26% |
| Translation | BLEU-4 | ~21-22 |

### View Results

```bash
# TensorBoard (real-time during training)
tensorboard --logdir=experiments/runs/exp_YYYYMMDD_HHMMSS/logs

# Static plots
ls experiments/runs/exp_YYYYMMDD_HHMMSS/logs/plots/
# - training_curves.png
# - train_val_comparison.png
# - training_summary.png
```

## 📊 Results

### Current Implementation Status

**Completed**:
- ✅ Data pipeline (7,096 training samples)
- ✅ Model architecture (SLRT + SLTT)
- ✅ Joint training system
- ✅ Loss functions (CTC + CrossEntropy)
- ✅ Training loop with validation
- ✅ Visualization and logging

**In Progress**:
- 🔄 Model convergence optimization
- 🔄 Hyperparameter tuning
- 🔄 Full training to target performance

**Known Issues**:
- Training loss decreasing slowly (investigating CNN features)
- Need pretrained sign language features (currently using ImageNet)

### Sample Training Output

After 100 epochs:
- Combined Loss: ~33.8 (train), ~33.0 (val)
- Recognition Loss: ~5.95 (train), ~5.78 (val)
- Translation Loss: ~4.05 (train), ~4.02 (val)

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Error**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 2  # or 1
```

**2. Windows Multiprocessing Error**
```yaml
# Set num_workers to 0
training:
  num_workers: 0
```

**3. CUDA Not Available**
```yaml
# Use CPU
training:
  device: cpu
```

**4. Import Errors**
```bash
# Ensure you're in project root
cd /path/to/sign-language-transformers
python scripts/05_train.py
```

**5. Slow Training**
- Use GPU if available
- Reduce max_frames in dataset (32 → 16)
- Increase batch_size if memory allows

### Debug Mode

Add to `config.yaml` for debugging:
```yaml
training:
  batch_size: 1
  num_epochs: 2
  num_workers: 0
```

## 📝 Model Checkpoints

Models are saved in:
```
experiments/runs/exp_YYYYMMDD_HHMMSS/checkpoints/
├── best_model.pth           # Best validation loss
└── checkpoint_epoch_XXX.pth # Every epoch
```

### Load Checkpoint

```python
import torch
from src.models.joint_model import SignLanguageTransformer

# Create model
model = SignLanguageTransformer(gloss_vocab_size=1811, word_vocab_size=2897)

# Load checkpoint
checkpoint = torch.load('experiments/runs/.../checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🔬 Architecture Details

### Sign Language Recognition Transformer (SLRT)

**Input**: Video frames (batch, time, 260, 210, 3)

**Components**:
1. CNN: Extract spatial features (260×210×3 → 1024)
2. Transformer Encoder (3 layers): Learn temporal dependencies
3. CTC Classifier: Predict glosses (→ 1812 classes)

**Output**: CTC log probabilities (batch, time, vocab_size)

### Sign Language Translation Transformer (SLTT)

**Input**: Encoder output + target words

**Components**:
1. Word Embedding: Convert words to vectors (2897 → 512)
2. Transformer Decoder (3 layers): Autoregressive generation
3. Output Projection: Predict next word (512 → 2897)

**Output**: Translation logits (batch, seq_len, vocab_size)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{camgoz2020sign,
  title={Sign language transformers: Joint end-to-end sign language recognition and translation},
  author={Camg{\"o}z, Necati Cihan and Koller, Oscar and Hadfield, Simon and Bowden, Richard},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10023--10033},
  year={2020}
}
```

## 🙏 Acknowledgments

- Original paper: Camgöz et al., CVPR 2020
- PHOENIX-2014-T dataset: RWTH Aachen University
- Implementation framework: PyTorch
- Visualization: TensorBoard, Matplotlib, Seaborn

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## 🔗 Useful Links

- [Original Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf)
- [PHOENIX-2014-T Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)

---

**Status**: 🚧 In Active Development

Last Updated: January 3, 2025