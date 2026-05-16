# Enzyme Kinetic Parameters Prediction 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)

This project provides a robust, multimodal deep learning framework to accurately predict *in vitro* enzyme kinetic parameters—specifically turnover number ($k_{cat}$), Michaelis constant ($K_m$), and inhibition constant ($K_i$)—given an enzyme-substrate pair.

Inspired by state-of-the-art architectures like **CatPred**, **UniKP**, and **ERBA**, this repository implements a streamlined bimodal feature fusion model (Sequence + Substrate) and rigorously evaluates its generalization capabilities on Out-of-Distribution (OOD) test sets based on sequence homology.

---

## 🌟 Key Features

1. **Multimodal Feature Fusion**:
   * **Enzyme Sequence**: Utilizes pre-trained Protein Language Models (e.g., `ESM-2`) to extract deep biochemical and evolutionary semantic features.
   * **Substrate Molecule**: Employs pre-trained Chemical Language Models (e.g., `ChemBERTa`) or Graph Neural Networks to process canonical SMILES strings.
2. **Heteroscedastic Uncertainty Quantification**:
   * Optimizes a Heteroscedastic Gaussian Negative Log-Likelihood (NLL) loss. The model outputs both the predicted mean ($\mu$) and variance ($\sigma^2$), effectively quantifying the aleatoric uncertainty inherent in noisy experimental data.
3. **Strict Out-of-Distribution (OOD) Evaluation**:
   * Abandons traditional random splitting in favor of pre-computed, sequence-similarity-based splits (e.g., 99%, 80%, 60%, 40% identity thresholds) to truthfully assess the model's performance on unseen enzyme sequences.

---

## 📂 Directory Structure

```text
EKP_Prediction_Project/
│
datasets/
datasets/
├── processed/                                # Master dataset files
│   ├── kcat_max_wt_singleSeqs_wpdbs.csv      # ~23k data points [cite: 280]
│   ├── km_mean_wt_singleSeqs_wpdbs.csv       # ~41k data points [cite: 280]
│   └── ki_mean_wt_singleSeqs_wpdbs.csv       # ~12k data points [cite: 280]
├── splits/                                   # All split masks stored flat
│   # --- Turnover Number (kcat) Splits ---
│   ├── kcat-random_train.csv                 # In-distribution training set mask
│   ├── kcat-random_val.csv                   # In-distribution validation set mask
│   ├── kcat-random_test.csv                  # In-distribution test set mask
│   ├── kcat-seq_test_sequence_99cluster.csv  # OOD test: max 99% identity to train [cite: 3655, 3680]
│   ├── kcat-seq_test_sequence_80cluster.csv  # OOD test: max 80% identity to train [cite: 3680]
│   ├── kcat-seq_test_sequence_60cluster.csv  # OOD test: max 60% identity to train [cite: 3680]
│   ├── kcat-seq_test_sequence_40cluster.csv  # OOD test: max 40% identity to train [cite: 3680]
│   # --- Michaelis Constant (Km) Splits ---
│   ├── km-random_train.csv                   # In-distribution training set mask
│   ├── km-random_val.csv                     # In-distribution validation set mask
│   ├── km-random_test.csv                    # In-distribution test set mask
│   ├── km-seq_test_sequence_99cluster.csv    # OOD test: max 99% identity to train [cite: 3680]
│   ...
│   # --- Inhibition Constant (Ki) Splits ---
│   ├── ki-random_train.csv                   # In-distribution training set mask
│   ├── ki-random_val.csv                     # In-distribution validation set mask
│   └── ki-seq_test_sequence_40cluster.csv    # OOD test: max 40% identity to train [cite: 3680]
│
├── notebooks/                  # Jupyter Notebooks for EDA and prototyping
│   └── 01_data_exploration.ipynb
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset and DataLoader construction
│   ├── encoders.py             # Feature extractors (ESM-2, ChemBERTa)
│   ├── models.py               # Multimodal fusion and dual-head regression network
│   ├── losses.py               # Custom uncertainty-aware loss functions
│   └── metrics.py              # Evaluation metrics (R^2, PCC, RMSE, MAE, p1mag)
│
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation and OOD testing script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation (this file)

