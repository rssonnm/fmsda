# Flow Matching for Spectral Data Augmentation

This is a state-of-the-art generative model designed to augment 1D spectral data (specifically UV-Vis absorbance spectra of citrus varieties). It leverages **Conditional Flow Matching (CFM)** with **Optimal Transport (OT)** paths, significantly outperforming traditional GANs in stability and sample quality.

The project incorporates advanced techniques from modern generative modeling (Stability AI, OpenAI) adapted for scientific 1D data:
- **Optimal Transport Conditional Flow Matching (OT-CFM)** for stable training.
- **Diffusion Transformer (DiT-1D)** backbone with **AdaLN-Zero** conditioning.
- **2-Rectified Flow (Reflow)** for straight-line probability paths enabling 1-step or few-step generation.
- **Classifier-Free Guidance (CFG)** for controllable generation strength.

Optimized for **Apple Silicon (MPS)**, CUDA, and CPU.

---

## Key Features

-   **Architecture**:
    -   **DiT-1D**: A Transformer specialized for 1D sequences, using patch embeddings and adaptive layer normalization.
    -   **VelocityNetwork (MLP)**: A lightweight alternative for faster training on small datasets.
-   **Training Objective**:
    -   Minimizes the regression loss between the model's predicted velocity and the target vector field constructed via Optimal Transport.
-   **Sampling**:
    -   Supports **Euler** and **RK4** ODE solvers.
    -   **Reflow** enables high-quality sampling with as few as 1-3 steps.
-   **Data Processing**:
    -   Automatic **Savitzky-Golay** derivative calculation to focus on spectral shape (peaks) rather than baseline intensity.

---

## Project Structure

```
fmsda/
├── src/
│   └── fm_sda/
│       ├── __init__.py
│       ├── models.py       # DiT-1D, VelocityNetwork, AdaLN
│       ├── engine.py       # FlowMatchingTrainer, Reflow Logic
│       ├── samplers.py     # ODE Solvers (Euler, RK4), CFG
│       └── data.py         # SpectralDataset, Preprocessing
├── scripts/
│   ├── train.py                # Main training script (OT-CFM)
│   ├── generate_reflow_data.py # Generate (z0, z1) pairs for Reflow
│   ├── train_reflow.py         # Train Reflow (Distillation) model
│   └── visualize.py            # Generate plots (Spectra, PCA, t-SNE)
├── data/
│   └── processed/              # Stores generated Reflow data
├── results/                    # Output plots and figures
└── checkpoints/                # Saved models
```

---

## Installation

Ensure you have a Python environment (Python 3.10+ recommended).

1.  **Clone the repository** (if applicable).
2.  **Install Dependencies**:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn pandas tqdm
```

*Note: For Apple Silicon acceleration, ensure you have the `mps` enabled version of PyTorch.*

---

## Usage

### 1. Train Standard Model (OT-CFM)

Train the base model (Teacher) using Optimal Transport Conditional Flow Matching.

```bash
python scripts/train.py \
    --data_path "../path/to/data.csv" \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --sigma_min 1e-4 \
    --derivative    # Apply 1st derivative
```

### 2. Reflow (Optional but Recommended)

"Reflow" distills the model to straighten the generation paths, improving quality and speed.

**Step A: Generate Reflow Dataset**
Use the trained base model to generate pairs $(z_0, z_1)$.

```bash
python scripts/generate_reflow_data.py \
    --checkpoint checkpoints/fm_sda_model.pth \
    --save_dir data/processed \
    --num_samples 50000 \
    --steps 50
```

**Step B: Train Reflow Model (Student)**
Train a DiT-1D model on the generated pairs with Classifier-Free Guidance support.

```bash
python scripts/train_reflow.py \
    --data_path data/processed/reflow_data.pt \
    --use_dit \
    --epochs 100 \
    --p_uncond 0.1 \
    --save_dir checkpoints_reflow
```

### 3. Visualization & Evaluation

Generate spectral plots, PCA, and t-SNE comparisons. Use `--cfg_sweep` to analyze the effect of guidance.

```bash
python scripts/visualize.py \
    --checkpoint checkpoints_reflow/reflow_model.pth \
    --data_path "../path/to/data.csv" \
    --results_dir results_reflow \
    --cfg_sweep \
    --steps 20
```

---

## Results

The model generates high-fidelity spectra that statistically match the real data distribution.

-   **Spectral Consistency**: Mean and Std of generated spectra overlap perfectly with real data.
-   **Manifold Learning**: PCA and t-SNE plots show generated data occupying the same manifold as real samples.

*(See `results/` directory for generated plots)*

---
