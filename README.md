# AI-for-Identifying-the-Origin-of-Scattered-Gamma-Rays---GammaTPC
A research repository exploring advanced Analytical, Deep Learning, and Hybrid methods for reconstructing low-energy electron recoil tracks in gas-phase Time Projection Chambers (TPC), specifically for gamma-ray polarimetry applications.

**Core Methodologies**
**1. Unified Sparse 3D U-Net (Deep Learning) A specialized Deep Learning pipeline processing sparse 3D point clouds to accurately predict the electron interaction origin and initial direction.
**
Architecture: SpConv-based 3D U-Net backbone with multi-task regression heads.
Technique: Utilizes Weighted Soft-ArgMax (Integral Regression) for sub-voxel origin localization and Transformer-based Pooling for direction estimation.
Performance: Achieved state-of-the-art origin precision with <1.0 mm median error on high-diffusion tracks.
(AI_model)

**2. Scored Endpoint Segment PCA (SES-PCA): A purely analytical baseline algorithm developed to reconstruct tracks without training data.**

Logic: Identifies track extremities using a Double-Farthest-Point search.
Scoring: Evaluates "Head" vs "Tail" candidates using a multi-metric scoring engine (Charge Density, Local Linearity, and Curvature) before applying PCA for final vector estimation.
(Analytical method)

**3. Geometric-Adaptive Kalman Voting (Hybrid) A post-processing module designed to resolve directionality in complex scattering scenarios.**

Logic: Combines Kalman Filter smoothing with a robust Start Selector heuristic.
Voting: Uses a density-adaptive voting mechanism to dynamically select between the smooth Kalman Tangent and robust Secant vectors.
(Directory Termius/direction_standalone.py + KF playground for experimenting)
