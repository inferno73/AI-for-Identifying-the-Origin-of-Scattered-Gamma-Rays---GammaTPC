# AI Model Specification: "GammaTPC Unified Sparse U-Net"

This document provides details on the neural network architecture, data flow, and training mechanisms for the proposed direction reconstruction model.

## 1. High-Level Architecture
The model is a **Multi-Task Sparse 3D U-Net**. It processes the entire sparse point cloud of a track at once and predicts two distinct physical properties via separate "Heads":
1.  **Origin Head**: Predicts the true vertex location $(x,y,z)$ to correct the initial noisy estimate.
2.  **Direction Head**: Predicts the initial direction vector $(\hat{v}_x, \hat{v}_y, \hat{v}_z)$ of the primary particle.

The backbone is a shared Sparse ResNet encoder-decoder that learns hierarchical features common to both tasks (e.g., Bragg peak identification, topology classification).

---

## 2. Input Data Specification
The model accepts a **Sparse Tensor** defined by coordinates and features.

*   **Coordinates**: $(N, 4)$ matrix of $[BatchID, x, y, z]$.
    *   $x, y, z$ are quantized voxel coordinates (e.g., scale 1mm).
    *   The cloud is centered such that the *approximate* origin is at $(0,0,0)$.
*   **Features**: $(N, C_{in})$ matrix.
    *   **V1 Implementation**: $C_{in}=1$ (Charge).
    *   **V2 Enhancement**: $C_{in}=4$ (Charge, dQ_local, Linearity, Gradient).
*   **Batching**: Variable number of points $N$ per track. Processed using sparse collation (MinkowskiEngine/spconv style).

---

## 3. Network Architecture components

### A. Shared Backbone (Sparse U-Net)
*   **Encoder**: 4 Levels of [Conv3D(stride=2) -> ResBlock -> ResBlock].
    *   Downsamples spatial resolution by factor of $2^4 = 16$.
    *   Increases channel depth (e.g., 16 -> 32 -> 64 -> 128).
*   **Bottleneck**: Deepest layer capturing global topology (e.g., "is this a V-shape pair or a line Compton?").
*   **Decoder**: 4 Levels of [ConvTranspose3D(stride=2) -> Concat(Skip) -> ResBlock].
    *   Restores full spatial resolution.

### B. Head 1: Origin Refinement
*   **Input**: Full resolution feature map from Decoder.
*   **Structure**: `1x1 Conv` -> `Sigmoid`.
*   **Output**: A heatmap $H \in \mathbb{R}^{N \times 1}$ representing the probability of each voxel being the true origin.
*   **Readout**:
    *   **Soft-Argmax (DSNT)**: Differentiable center-of-mass calculation.
    *   $\text{Pred}_{offset} = \frac{\sum H_i \cdot x_i}{\sum H_i}$.
    *   The predicted origin correction is this offset.

### C. Head 2: Direction Field & Transformer Pooling
*   **Input**: Full resolution feature map from Decoder.
*   **Structure**:
    1.  **Voxel Predictor**: `1x1 Conv` -> Output $(v_x, v_y, v_z)$ per voxel.
    2.  **Ring Masking**: Only voxels within radius $R$ (e.g., 10mm) of the specific origin contribute.
*   **Pooling Mechanism (Enhancement)**:
    *   Takes the feature vectors $F_i$ of all valid voxels.
    *   **Transformer Block**: A small Self-Attention layer ($L=1$, Heads=4) processes the sequence of features.
    *   **Global Average**: The transformed features are averaged to produce the final global direction vector $\vec{V}_{final}$.
    *   This allows the model to "weigh" the Bragg peak vs. multiple scattering tails intelligently.

---

## 4. Loss Functions & Training

### Origin Loss ($\mathcal{L}_{org}$)
MSE between the predicted offset and the true label offset.
$$ \mathcal{L}_{org} = || \text{Pred}_{offset} - \text{Label}_{offset} ||^2 $$

### Direction Loss ($\mathcal{L}_{dir}$)
Cosine similarity loss, but crucially applied with **Ring Supervision**.
$$ \mathcal{L}_{dir} = 1 - \text{CosineSimilarity}(\vec{V}_{final}, \vec{V}_{true}) $$
*Note*: Can also supervise per-voxel vectors to align with local tangent.

### Total Loss
$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{org} + \lambda_2 \mathcal{L}_{dir} $$
(Typically $\lambda_1=1.0, \lambda_2=1.0$ initially).

---

## 5. Implementation Stack
*   **Framework**: PyTorch.
*   **Sparse Backend**: `MinkowskiEngine` (preferred for Linux) or `torchsparse` / `spconv` (Windows alternatives).
*   **Note on Windows**: Since we are developing on Windows locally, standard `spconv` is the most reliable sparse library. If unavailable, we can write a "Dense Mock" for debugging that treats the sparse cloud as a dense 3D grid, then switch to sparse for production.
