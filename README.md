# Multi-Modal Heteroscedastic Regression for OOD Enzyme Kinetics Prediction

This repository contains the official implementation of a probabilistic, multi-modal deep learning framework engineered for the zero-shot out-of-distribution (OOD) prediction of fundamental enzyme kinetic parameters ($k_{\text{cat}}$, $K_m$, and $K_i$) across sequence-disjoint wilds.

By departing from standard homoscedastic regression assumptions, the model implements a specialized **Gaussian Twin-Head Architecture** optimized via Negative Log-Likelihood (NLL) loss to dynamically quantify and down-weight data-dependent experimental noise (heteroscedasticity) inherent to multi-laboratory assay datasets.

---

## 🚀 Key Features

* **Probabilistic Heteroscedastic Head**: Replaces naive MSE loss with Gaussian Negative Log-Likelihood optimization, producing both the predictive mean $\mu(X)$ and data-dependent uncertainty $\sigma^2(X)$ to naturally isolate experimental outliers.
* **3x3 Progressive Ablation Matrix**: Implements a strict benchmarking grid crossing 3 kinetic endpoints ($k_{\text{cat}}$, $K_m$, $K_i$) and 3 evolutionary scales:
  * **Sequence-Only**: Pre-trained Protein Language Model ESM-2 (`esm2_t6_8M_UR50D`).
  * **Bimodal**: Sequence + Substrate/Inhibitor Graph Transformer ChemBERTa (`ChemBERTa-77M-MTR`).
  * **Trimodal**: Bimodal + Stereochemical 3D Proxy equipped with an MD5 cryptographic fallback manager.
* **Unified Backbone Production Pipeline**: High-efficiency inference engine that loads heavy foundation transformers into GPU memory exactly once, treading across 9 independent checkpoints via programmatic weight-swapping.
* **Automated Scientific Dashboard**: One-click generation of 31 publication-quality diagnostic visualizations (3x3 regression scatters, residual diagnostics, multi-dimensional radars, and error trend sheets).

---

## 📂 Repository Structure

```text
├── checkpoints/               # Directory containing 9 trained model checkpoints (.pt)
├── pictures/                  # Auto-generated visualization dashboard (.png output)
├── data/                      # Data repository containing master records and splitting masks
├── train.py                   # Central script for network training and loss minimization
├── evaluate.py                # High-efficiency batch inference and diagnostic visualization engine
├── requirements.txt           # Environment dependencies list
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

1. Clone this repository and navigate into the root directory:
   ```bash
   git clone [https://github.com/your-username/enzyme-kinetics-multimodal.git](https://github.com/your-username/enzyme-kinetics-multimodal.git)
   cd enzyme-kinetics-multimodal
   ```

2. Install all required dependencies via pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your environment has access to the Hugging Face Hub for automatic model backbone caching (`facebook/esm2_t6_8M_UR50D` and `DeepChem/ChemBERTa-77M-MTR`).

---

## 💻 Usage Guide

### 1. Training Individual Matrix Cells
Open `train.py` and adjust the central control switch panel to your target combination:

```python
# train.py Central Switch Panel
TARGET_TASK = "km"         # Options: "kcat", "km", "ki"
MODEL_MODE  = "bimodal"    # Options: "seq_only", "bimodal", "trimodal"
```

Then fire up the training execution pipeline:
```bash
python train.py
```

The script will automatically run for 20 epochs, monitor evaluation metrics, and dump the optimal checkpoint into the `checkpoints/` repository using our defensive anti-overwrite filename pattern.

### 2. High-Efficiency Batch Evaluation & Visualization
You no longer need to manually modify target configurations for inference. The re-engineered single-backbone batch pipeline will automatically loop through all 3 tasks $\times$ 3 modalities, load foundation backbones once, switch model weights seamlessly, and dump 31 diagnostic charts into `pictures/`:

```bash
python evaluate.py
```

---

## 📊 OOD Validation Highlights (Sequence-Disjoint Wilds)

When evaluated under strict sequence-disjoint constraints (where test protein families share less than 40% sequence identity with the training set), single-modality baseline trackers encounter catastrophic generalization collapse. Multi-modal alignment anchors the model's predictive capacity:

* **Binding Affinity Quantum Leaps**: Integrating sub-structural graph languages pushes the OOD $R^2$ from negative baselines up to **0.4058** ($K_m$) and **0.3881** ($K_i$), with Pearson Correlation Coefficients (PCC) soaring past **0.64**.
* **Catalytic Constant Volatility**: Tracking catalytic turnover rates ($k_{\text{cat}}$) highlights a structural heterogeneity, demonstrating high sensitivity to global evolutionary folds governed by transition-state thermodynamics rather than static chemical graphs.

---

## 📑 License & Citations
This project is developed as part of the *Statistical Learning and Data Science Project Design* curriculum at the School of Mathematics, South China University of Technology.

For method details, mathematical derivations of the heteroscedastic NLL objective, or biochemical discussions regarding transition-state stabilization, please refer to our final project report.
```