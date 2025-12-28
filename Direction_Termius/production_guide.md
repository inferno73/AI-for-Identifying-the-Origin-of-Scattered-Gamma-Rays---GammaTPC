# Hybrid Direction Pipeline - Termius Deployment

This folder contains the "Universal Algorithm" (Run 9) adapted for the Remote Model Evaluation loop.

## 1. Files
*   `evaluate_kf_hybrid.py`: Main script. Evaluates model + runs Hybrid Direction logic on predictions.
*   `kalman_tracking.py`: KF implementation (Dependency).
*   `train_unified_kf.py`: Model definition (Dependency).

## 2. Setup (Termius)
Upload **only** `evaluate_kf_hybrid.py` into your existing model folder `1.model_KF`.
(The other dependencies `kalman_tracking.py` and `train_unified_kf.py` should already be there).

Structure:
```
AI_playground/
  ├── dataset_test.npz
  └── 1.model_KF/
        ├── model_best.pth
        ├── evaluate_kf.py (Old)
        ├── evaluate_kf_hybrid.py  <-- NEW (Run this)
        ├── kalman_tracking.py
        └── train_unified_kf.py
```

## 3. Execution
Run the evaluation script from within the `1.model_KF` folder.

### Command
```bash
cd 1.model_KF
python evaluate_kf_hybrid.py \
  --dataset_dir "../"
```
*(No need to specify `--model_path` if `model_best.pth` is in the same folder).*

### Parameters
*   `--dataset_dir`: Path containing `dataset_test.npz` (e.g. `../` if inside model folder).
*   `--model_path`: Optional. Defaults to `model_best.pth` or `model_best_origin.pth` in current dir.

### Command
```bash
cd 1.model_KF
python evaluate_kf_hybrid.py \
  --dataset_dir "../"
```

## 5. Alternative: Standalone Direction Testing (With or Without Model)
This script tests the Direction Algorithm on the RAW dataset files.
You can run it in two modes:
1.  **Ideal Mode (Cheating)**: Uses True Origin to test the Algorithm's theoretical limit.
2.  **Real Mode (Model)**: Uses `model_best.pth` to predict Origin, then runs Algorithm.

### Setup
Upload THREE files to `1.model_KF`:
*   `test_direction_standalone.py`
*   `kalman_tracking.py`
*   `train_unified_kf.py` (REQUIRED if using model)

### Command (Job Submission)
Since the defaults are now set to your specific paths, you can simply run:
```bash
./fast_bg_exec.sh python test_direction_standalone.py
```

However, to be explicit (or if paths change), use:
```bash
./fast_bg_exec.sh python test_direction_standalone.py \
  --dataset_dir "/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks906/" \
  --output_dir "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground/1.model_KF/results_FULL" \
  --model_path "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground/1.model_KF/model_best.pth"
```

**Output**:
*   Report: `standalone_report_FULL.txt` inside `results_FULL/`
*   Plots: `results_FULL/analysis_plots/*.html` (Best 30 and Worst 30)
