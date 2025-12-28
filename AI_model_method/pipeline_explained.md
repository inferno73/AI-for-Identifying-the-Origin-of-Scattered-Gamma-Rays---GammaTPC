# Multi-Head Sparse U-Net Pipeline for GammaTPC

This document outlines the improved AI model approach for determining the origin and direction of particle tracks in the GammaTPC project. It builds upon the mentor's baseline (Sparse CNN + Ring Supervision) but introduces a **Unified Multi-Task Architecture** and **Transformer-Based Pooling** for enhanced precision.

## 1. Pipeline Overview

The pipeline consists of three main stages:

1.  **Preprocessing**: Converting `.npz` track data into a sparse, origin-centered dataset with physics-based features.
2.  **Training**: A single Multi-Head Sparse U-Net that simultaneously learns to pinpoint the origin and predict the direction.
3.  **Inference**: A unified evaluation script that outputs both origin coordinates (mm) and direction vectors (degrees) for final analysis.

--------------------------------------------------------------------------------

## 2. Component Details

### Stage 1: Dataset Preparation (`make_dataset.py`)

**Goal**: Create a unified dataset that works for both tasks.

*   **Input**: `compton/*.npz` and `pair/*.npz`
*   **Features per Voxel**:
    *   `charge`: Raw deposited charge.
    *   `dQ_local`: Charge contrast with neighbors.
    *   `linearity`: Local PCA linearity score.
    *   `grad_xyz`: Local gradient vector.
*   **Augmentation**:
    *   **Gaussian Origin Noise**: Perturb the true origin by sigma approx 2-5mm to simulate detector uncertainty.
    *   **Centering**: Shift cloud so the *noisy* origin is at `(0,0,0)`.
*   **Output**: Compressed sparse tensors (`coords`, `features`, `labels`).

### Stage 2: The Multi-Head Sparse U-Net (`train_unified.py`)

**Architecture**: A single `MinkowskiEngine` or `spconv` based 3D U-Net backbone with a shared encoder and two specialized decoders/heads.

#### Backbone (Shared Encoder)
*   **Input**: Sparse Tensor `(N, C_in)`.
*   **Layers**: 4-5 stages of Sparse ResNet Blocks with stride-2 convolutions.
*   **Function**: Extracts hierarchical features (from local dE/dx clusters to global V-shape topology).

#### Head A: Origin Refinement (Regression)
*   **Task**: Correct the initial noisy origin.
*   **Mechanism**: A lightweight decoder that outputs a per-voxel "Origin Probability" heatmap.
*   **Output layer**: DSNT (Soft-Argmax) or simple weighted average of top-k voxels to predict the `(dx, dy, dz)` offset.
*   **Loss**: `L_origin = || Pred_pos - True_origin ||^2`.

#### Head B: Direction Prediction (Vector Field)
*   **Task**: Predict the initial tangent vector of the particle.
*   **Mechanism**:
    1.  **Voxel Prediction**: Each voxel in the valid "Ring" predicts a unit vector `v_i`.
    2.  **Transformer Pooling (New Improvement)**:
        *   Instead of a simple weighted average, active voxels within the "Ring of Interest" (ROI) are treated as tokens.
        *   A small **Self-Attention Layer** (Transformer Block) processes these tokens to understand the *sequence* of the track (Bragg peak evolution).
        *   Output: A single global direction vector `V_final`.
*   **Loss**: `L_dir = 1 - cos(theta)`, applied only to voxels in the Gaussian Ring (Ring Supervision).

#### Global Loss
`L_total = lambda_1 * L_origin + lambda_2 * L_dir`

--------------------------------------------------------------------------------

## 3. Implementation Steps

We will proceed in the following order:

1.  **`make_direction_dataset.py`**: Implement the data loader and feature extractor.
2.  **`model.py`**: Define the Unified Sparse U-Net class with Transformer pooling.
3.  **`train.py`**: The main training loop with the combined loss function.
4.  **`evaluate.py`**: Validation script to measure Origin Error (mm) and Angle Error (degrees).

## 4. Key Improvements over Baseline

| Feature | Mentor's Pipeline | Improved Pipeline |
| :--- | :--- | :--- |
| **Task Handling** | Separate models for Origin & Direction | **Single Multi-Task Model** (Faster, regularized) |
| **Direction Pooling** | Attention Weighted Average | **Transformer/Self-Attention** (Captures dependencies) |
| **Data Flow** | 2-Pass (Find origin -> Crop -> Find dir) | **1-Pass** (Refine origin & Find dir simultaneously) |
