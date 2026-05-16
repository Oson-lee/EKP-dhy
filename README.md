# Enzyme Kinetic Parameters Prediction: A Multimodal Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-orange.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)
![CUDA](https://img.shields.io/badge/CUDA-12.6-success)

This project provides a robust, multimodal deep learning framework to accurately predict *in vitro* enzyme kinetic parameters—specifically turnover number (kcat), Michaelis constant (Km), and inhibition constant (Ki)—given an enzyme-substrate pair. 

Developed as a course project for *Introduction to Statistic Learning and Data Science*, this repository implements a streamlined bimodal feature fusion model (Protein Sequence + Substrate SMILES) and rigorously evaluates its generalization capabilities on Out-of-Distribution (OOD) test sets based on sequence homology.

---

## 🌟 Key Features

1. **Multimodal Feature Fusion Network**:
   * **Enzyme Sequence**: Utilizes pre-trained Protein Language Models (`facebook/esm2_t6_8M_UR50D`) to extract deep biochemical and evolutionary semantic features.
   * **Substrate Molecule**: Employs pre-trained Chemical Language Models (`DeepChem/ChemBERTa-77M-MTR`) to process canonical SMILES strings into dense embeddings.
2. **Heteroscedastic Uncertainty Quantification**:
   * Optimizes a custom **Heteroscedastic Gaussian Negative Log-Likelihood (NLL) loss**. The model outputs both the predicted mean (μ) and log-variance (log σ²), effectively quantifying the aleatoric uncertainty inherent in noisy experimental biochemical data.
3. **Strict Out-of-Distribution (OOD) Evaluation**:
   * Abandons traditional random splitting in favor of pre-computed, sequence-similarity-based splits. The final model is evaluated on a rigorous **< 40% sequence identity** threshold to truthfully assess its performance on completely unseen, non-homologous enzyme families.
4. **Adaptive Training Pipeline**:
   * Features dynamic gradient clipping and a `ReduceLROnPlateau` scheduler to ensure training stability and prevent gradient explosion when optimizing exponential variance parameters.

---

## 🏆 Project Results (OOD Benchmark)

The model was evaluated on the most rigorous OOD split (`kcat-seq_test_sequence_40cluster.csv`), ensuring no enzyme in the test set shares more than 40% sequence identity with the training data.

**Final OOD Test Set Performance (kcat):**

* **R-Squared (R²)**: 0.2815
* **Pearson Correlation**: 0.5313
* **RMSE**: 1.4244
* **MAE**: 1.0592

*Note: Achieving an R² > 0.25 and Pearson > 0.5 on a 40% cluster split using frozen base encoders demonstrates strong baseline generalization and successful multimodal alignment.*

---

## 📂 Directory Structure

```text
EKP_Prediction_Project/
├── datasets/                                 
│   ├── processed/                            # Master dataset files (~23k kcat data points)
│   └── splits/                               # Training and OOD testing masks
├── checkpoints/                              # Saved model weights
│   └── best_multimodal_model.pt              # Auto-saved best weights based on Val R²
├── src/                                      # Core source code
│   ├── __init__.py
│   ├── dataset.py                            # PyTorch Dataset and DataLoader construction
│   ├── encoders.py                           # Feature extractors (ESM-2, ChemBERTa)
│   ├── models.py                             # Multimodal fusion and dual-head regression MLP
│   ├── losses.py                             # Heteroscedastic Gaussian NLL Loss
│   └── metrics.py                            # Evaluation metrics (R², Pearson, RMSE, MAE)
├── train.py                                  # Main training script with LR scheduling
├── evaluate.py                               # OOD inference and evaluation script
└── README.md                                 # Project documentation
```

## 🚀 Installation & Setup

**1. Clone the repository and navigate to the root directory:**

```
git clone <your-repo-url>
cd EKP_Prediction_Project
```

**2. Install dependencies:** Ensure you have a CUDA-capable GPU (e.g., RTX 3060) for accelerated training. Install the specific PyTorch version with CUDA support:

```
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
pip install transformers scikit-learn scipy pandas numpy tqdm
```

------

## 💻 Usage

**Training the Model:** To train the multimodal fusion network from scratch, run:

```
python train.py
```

*The script will automatically detect CUDA, freeze the heavy LLM encoders, train the fusion MLP, and save the best weights to the `checkpoints/` directory.*

**Evaluating on OOD Data:** To test the trained model's generalization on the rigorous 40-cluster split, run:

```
python evaluate.py
```



Remain to be done:

```
EKP_Prediction_Project/
│
├── datasets/
│   ├── processed/                      # 存放 kcat, km, ki 的三种主数据表
│   ├── splits/                         # 存放所有参数的 train/val/ood 划分掩码
│   └── structures/                     # [新增] 满足要求3：3D结构数据
│       ├── pdb_files/                  # 存放从 AlphaFold 下载的 .pdb 原始文件
│       └── graphs/                     # 存放预处理好的 PyTorch Geometric 图数据 (.pt)
│
├── checkpoints/                        # [重构] 按模态和任务分类保存权重
│   ├── baseline_seq_only/              # [新增] 满足要求1：单模态对比权重
│   │   ├── kcat_seq_only.pt
│   │   └── km_seq_only.pt
│   ├── bimodal_seq_sub/                # 我们刚才跑通的双模态权重
│   │   ├── kcat_bimodal.pt
│   │   ├── km_bimodal.pt
│   │   └── ki_bimodal.pt
│   └── trimodal_seq_sub_struct/        # [新增] 满足要求3：三模态(含3D结构)权重
│       └── kcat_trimodal.pt
│
├── src/                                # 核心驱动层
│   ├── __init__.py
│   ├── dataset.py                      # [待修改] 需增加读取 3D 图数据 (Graph) 的逻辑
│   ├── encoders.py                     # [待修改] 增加 StructureEncoder (3D GNN 提取器)
│   ├── models.py                       # [待修改] 增加 SequenceOnlyModel 和 TriModalModel
│   ├── losses.py                       
│   └── metrics.py                      
│
├── scripts_preprocess/                 # [新增] 数据预处理脚本
│   └── build_3d_graphs.py              # 将 PDB 转化为氨基酸接触图 (Adjacency Matrix)
│
├── train.py                            # [重构] 满足要求2：全能训练调度器
├── evaluate.py                         # [重构] 全能评估器
└── README.md
```

