# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **pFL-FOOGD (Personalized Federated Learning with Feature-Oriented Out-of-Distribution Generalization and Detection)** framework for marine plankton image recognition. The system combines:

- **FedRoD (Federated Robust Decoupling)**: Handles non-IID data across clients
- **FOOGD (Feature-Oriented Out-of-Distribution Generalization and Detection)**: Handles OOD detection and generalization

### Core Architecture

- **Backbone**: DenseNet-169/121 for feature extraction
- **FedRoD Model**: Dual-head architecture with:
  - `Head_G` (Generic head): Aggregated by server, learns universal knowledge
  - `Head_P` (Personal head): Kept locally on clients, learns client-specific biases
- **FOOGD Module**: Integrates two OOD components:
  - **SAG (Stein Augmented Generalization)**: Uses KSD Loss for feature space regularization
  - **SM3D (Score Matching)**: Trains lightweight scoring model for OOD detection

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Dataset Preparation
```bash
python split_dataset.py
```

### System Testing
```bash
python test_pipeline.py
```

### Federated Learning Training
```bash
python train_federated.py \
    --data_root ./Plankton_OOD_Dataset \
    --n_clients 5 \
    --alpha 5.0 \
    --communication_rounds 50 \
    --local_epochs 1 \
    --batch_size 64 \
    --model_type densenet121 \
    --use_foogd \
    --output_dir ./experiments_5.0
```

### Complete Training Parameters
```bash
python train_federated.py --help
```

Key parameters:
- `--data_root`: Dataset root directory (default: `./Plankton_OOD_Dataset`)
- `--n_clients`: Number of federated learning clients (default: 10)
- `--alpha`: Dirichlet distribution parameter for non-IID data partitioning (lower = more heterogeneous)
- `--communication_rounds`: Total federated learning rounds (default: 50)
- `--local_epochs`: Local training epochs per client (default: 1)
- `--batch_size`: Batch size for training (default: 32)
- `--client_fraction`: Fraction of clients selected per round (default: 1.0)
- `--model_type`: Backbone network (`densenet121` or `densenet169`)
- `--use_foogd`: Enable/disable FOOGD module (flag)
- `--image_size`: Input image size (default: 224)
- `--eval_frequency`: Evaluation frequency in rounds (default: 5)
- `--save_frequency`: Checkpoint save frequency (default: 10)
- `--device`: Training device (`cuda` or `cpu`, default: auto-detect)
- `--seed`: Random seed for reproducibility (default: 42)
- `--resume`: Resume training from latest checkpoint (flag)

## Code Architecture

### Core Modules

- **`models.py`**: Contains all neural network definitions
  - `DenseNetBackbone`: Feature extractor using DenseNet
  - `FedRoD_Model`: Dual-head architecture with generic and personal classifiers
  - `FOOGD_Module`: Integrates SAG and SM3D for OOD handling
  - `ScoreModel`: Lightweight network for score matching

- **`client.py`**: Federated learning client implementation
  - `FLClient`: Handles local training, model updates, and evaluation
  - Uses separate optimizers for main model and FOOGD components

- **`server.py`**: Federated learning server implementation
  - `FLServer`: Handles model aggregation and global updates
  - Manages both model and FOOGD module parameters

- **`data_utils.py`**: Data loading and federated data partitioning
  - Strict class definitions for ID (54), Near-OOD (26), and Far-OOD (12) categories
  - Dirichlet distribution for non-IID data partitioning

- **`train_federated.py`**: Main training script
  - Orchestrates federated learning rounds
  - Handles evaluation and checkpointing

### Data Structure and Categories

#### Strict Category Definitions (defined in `data_utils.py`)
- **ID Classes (54)**: Target labels for training/validation/testing (80/10/10 split)
  - Includes plankton species like Polychaeta types, Acartia sp., Calanoid types, etc.
  - Used for model training and in-distribution evaluation
- **Near-OOD Classes (26)**: Only for testing (OOD evaluation)
  - Related but distinct plankton categories (e.g., Polychaeta larva, Calanoid Nauplii)
  - Represents distribution shift within the same domain
- **Far-OOD Classes (12)**: Only for testing (OOD evaluation)
  - Unrelated categories (e.g., Fish egg, Bubbles, Particle types)
  - Represents significant domain shift

#### Dataset Directory Structure
```
./data/
    ├── ID_images/           # 54 classes (Train/Val/Test)
    │   ├── train/           # 80% of ID data
    │   ├── val/             # 10% of ID data
    │   └── test/            # 10% of ID data
    ├── OOD_Near/            # 26 classes (Test only)
    └── OOD_Far/             # 12 classes (Test only)
```

#### Data Preparation
```bash
# Run dataset splitting script (creates train/val/test splits)
python split_dataset.py

# The script will:
# 1. Create proper directory structure
# 2. Split ID classes into 80/10/10 train/val/test
# 3. Copy Near-OOD and Far-OOD classes to test directories
```

## Key Implementation Details

### Federated Learning Strategy
- **Communication Rounds**: 50-100 rounds (configurable via `--communication_rounds`)
- **Client Selection**: Fraction-based (`--client_fraction`) or full participation
- **Local Training**: 1-5 epochs per client (`--local_epochs`)
- **Aggregation**: Weighted averaging based on client sample sizes
- **Model Decoupling**: FedRoD separates generic (Head_G) and personal (Head_P) heads
  - Head_G aggregated by server, learns universal knowledge
  - Head_P kept locally, learns client-specific biases

### FOOGD Components (Feature-Oriented OOD Generalization & Detection)
- **SM3D (Score Matching)**: Lightweight scoring model for OOD detection
  - Combines Denoising Score Matching (DSM) and Maximum Mean Discrepancy (MMD)
  - Trained to distinguish ID vs OOD features
- **SAG (Stein Augmented Generalization)**: Feature space regularization
  - Uses Kernelized Stein Discrepancy (KSD) loss
  - Aligns feature distributions for better OOD generalization
- **OOD Detection**: Based on score model norm (higher norm = more likely OOD)
  - Computes AUROC and FPR95 for Near-OOD and Far-OOD separately

### Optimization and Training Flow
1. **Initialization**: Create FedRoD model with DenseNet backbone
2. **Client Setup**: Distribute non-IID data using Dirichlet distribution (`--alpha`)
3. **Federated Rounds**:
   - Server broadcasts global model (Head_G + backbone) to selected clients
   - Clients perform local training with both Head_G and Head_P
   - If `--use_foogd`: Train FOOGD components locally
   - Clients send updated Head_G to server
   - Server aggregates Head_G updates via weighted averaging
4. **Evaluation**: Periodic evaluation on ID test set and OOD detection

### Optimization Details
- **Main Model**: SGD optimizer with momentum (0.9) and weight decay (1e-4)
- **FOOGD Module**: Adam optimizer for score model training
- **Learning Rates**: Separate learning rates for main model and FOOGD components
- **Mixed Precision**: Supported via `torch.amp` for memory efficiency
- **Gradient Clipping**: Applied to prevent exploding gradients

## Evaluation Metrics

Training generates comprehensive evaluation including:
- **ID Accuracy**: Classification performance on in-distribution data
- **OOD Detection**: AUROC and FPR95 for Near-OOD and Far-OOD
- **Personalization Gain**: Difference between Head_P and Head_G accuracy
- **IN-C Accuracy**: Performance on corrupted/perturbed data

## Memory Considerations

- **GPU Memory**: Requires NVIDIA RTX 3060 (6GB VRAM) or higher
- **Batch Size**: Use 16-32 for memory-constrained environments
- **Model Selection**: DenseNet-121 for lower memory usage

## Troubleshooting

1. **Dataset Issues**: Ensure `split_dataset.py` has been run and data paths are correct
2. **Memory Errors**: Reduce batch size or switch to DenseNet-121
3. **Training Instability**: Adjust learning rate or increase local epochs
4. **Model Loading**: Check PyTorch version compatibility

## Experimental Output and Analysis

### Training Output Structure
Each experiment creates a timestamped directory with:
```
experiment_YYYYMMDD_HHMMSS/
├── config.json                    # Complete experiment configuration
├── training_history.json          # Training metrics per round
├── training_curves.png            # Loss and accuracy plots
├── checkpoints/                   # Model checkpoints
│   ├── round_10.pth
│   ├── round_20.pth
│   └── best_model.pth
└── final_evaluation/              # Comprehensive evaluation
    ├── evaluation_report.json     # All metrics (ID accuracy, OOD detection)
    ├── confusion_matrix.png       # ID classification confusion matrix
    ├── near_ood_detection.png     # Near-OOD detection curves (AUROC/FPR95)
    └── far_ood_detection.png      # Far-OOD detection curves (AUROC/FPR95)
```

### Experiment Analysis
```bash
# Visualize and compare multiple experiments
python visualize_experiments.py --experiment_dirs experiments/alpha0.1_no_foogd experiments/alpha0.1_with_foogd
```

### Key Evaluation Metrics
- **ID Accuracy**: Classification accuracy on in-distribution test data
- **OOD Detection (AUROC)**: Area Under ROC curve for Near-OOD and Far-OOD
- **OOD Detection (FPR95)**: False Positive Rate at 95% True Positive Rate
- **Personalization Gain**: Difference between Head_P (personal) and Head_G (generic) accuracy
- **Training Stability**: Loss convergence and accuracy trends across rounds