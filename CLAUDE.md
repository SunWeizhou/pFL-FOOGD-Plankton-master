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

### Key Parameters
- `--alpha`: Dirichlet distribution parameter (controls data heterogeneity)
- `--n_clients`: Number of federated learning clients (default: 10)
- `--use_foogd`: Enable/disable FOOGD module
- `--model_type`: Backbone network (`densenet121` or `densenet169`)

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

### Data Structure

Dataset follows strict category definitions:
- **ID Classes (54)**: Target labels for training/validation/testing (80/10/10 split)
- **Near-OOD Classes (26)**: Only for testing (OOD evaluation)
- **Far-OOD Classes (12)**: Only for testing (OOD evaluation)

Directory structure:
```
./data/
    ├── ID_images/           # 54 classes (Train/Val/Test)
    ├── OOD_Near/            # 26 classes (Test only)
    └── OOD_Far/             # 12 classes (Test only)
```

## Key Implementation Details

### Federated Learning Strategy
- **Communication Rounds**: 50-100 rounds
- **Client Selection**: Fraction-based or full participation
- **Local Training**: 1-5 epochs per client
- **Aggregation**: Weighted averaging based on sample sizes

### FOOGD Components
- **SM3D Loss**: Combines Denoising Score Matching (DSM) and Maximum Mean Discrepancy (MMD)
- **SAG Loss**: Uses Kernelized Stein Discrepancy (KSD) for feature space alignment
- **OOD Detection**: Based on score model norm (higher norm = more likely OOD)

### Optimization
- **Main Model**: SGD optimizer with momentum and weight decay
- **FOOGD Module**: Adam optimizer for score model training
- **Mixed Precision**: Supported via torch.amp

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

## Experimental Output

Training generates:
- Training curves (loss, accuracy, OOD metrics)
- Model checkpoints every 10 rounds
- Evaluation reports with confusion matrices
- Configuration and training history files