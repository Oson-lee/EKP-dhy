# 📊 统计学习与数据科学项目深度指南：酶动力学参数预测 (Advanced)

## 🎯 一、 项目核心定义

本项目旨在将酶动力学参数（$k_{\text{cat}}$、$K_m$、$K_i$）的预测问题转化为一个**多模态回归与特征对齐任务 (Multimodal Regression Task)**。系统需要整合蛋白质序列空间 ($S_e$)、底物化学空间 ($S_m$) 以及蛋白质 3D 结构空间 ($S_g$)，并解决跨酶家族与底物的分布偏移 (Distribution Shift) 问题。

## 🗺️ 二、 技术实现深度拆解 (Workflow Breakdown)

### 阶段 1：数据工程与分布探索 (Data Engineering & EDA)

真实世界的生化数据（如来自 BRENDA 或 SABIO-RK）充满噪声与矛盾，数据预处理是本项目的地基。

- [ ] **化学分子标准化 (Substrate Canonicalization)**：
  - 使用 `RDKit` 读取底物的 SMILES 字符串，将其转换为标准 (Canonical) SMILES，消除同一分子不同表示法造成的数据冗余。
  - 处理立体化学 (Stereochemistry) 标签与盐类/溶剂剥离。
- [ ] **蛋白质序列对齐与过滤 (Protein Sequence Mapping)**：
  - 将数据集中的酶与 `UniProt` 数据库进行交叉比对，确保氨基酸序列的正确性。
  - 移除含有非标准氨基酸（如 X, Z, B）的异常序列。
- [ ] **多重测量值的聚合 (Aggregation of Replicates)**：
  - 当同一个酶-底物对存在多个实验测量值时，需定义聚合策略（如取中位数、对数平均值），以降低实验误差带来的噪声。
- [ ] **目标变量转换 (Target Variable Transformation)**：
  - 酶动力学参数通常跨越多个数量级，必须对目标变量 $y$ 进行 $\log_{10}$ 转换，使其近似正态分布，避免极端值主导梯度下降的方向。
- [ ] **数据集统计分层**：按照酶委员会编号 (EC Number) 分析数据分布，确认四大类/六大类酶的样本分布是否极度不平衡。

### 阶段 2：多模态表征学习 (Multimodal Representation Learning)

此阶段的数学目标是学习三个映射函数，将离散的生物化学实体转换为连续的稠密向量 (Dense Vectors)。

- [ ] **模态 1：蛋白质序列编码器 (**$S_e \rightarrow H_e \in \mathbb{R}^{n \times d}$**)**
  - **实现方案**：冻结或微调大型蛋白质语言模型 (PLMs)，如 `ESM-2` (Evolutionary Scale Modeling) 或 `ProtT5`。
  - **特征提取**：提取最后一层隐藏状态，可选择使用 `[CLS]` token 作为全局表征，或对所有氨基酸 token 进行 Mean-Pooling。
- [ ] **模态 2：底物分子编码器 (**$S_m \rightarrow H_m \in \mathbb{R}^{m \times d}$**)**
  - **实现方案 A (2D Graph)**：将 SMILES 转换为分子图，使用图神经网络 (如 GCN, GIN, 或 MPNN) 学习原子与键结的拓扑结构。
  - **实现方案 B (1D Fingerprint)**：使用 RDKit 提取 Morgan Fingerprints (半径=2, 位元=2048)，通过 MLP 降维至 $d$ 维。
- [ ] **模态 3：3D 结构与口袋几何 (**$S_g \rightarrow H_g \in \mathbb{R}^{g \times d}$**) [进阶消融]**
  - **数据来源**：从 PDB 获取实验解析结构，或使用 `AlphaFold2`/`ESMFold` 进行预测。
  - **实现方案**：构建基于残基的 K-近邻图 (KNN-Graph) 或半径图，通过 3D 图卷积网络 (如 SchNet 或 E(n)-Equivariant GNNs) 提取活性位点 (Active Site) 的空间折叠特征。

### 阶段 3：模型架构设计与多模态融合 (Architecture & Fusion)

模型需设计融合机制 $z = f(H_e, H_m, H_g)$ 来捕捉酶与底物之间的交互作用。建议设计三组对照实验 (Ablation Study)：

- [ ] **基准网络 (Baseline: Sequence-only)**
  - 仅依赖 $H_e$，将模型退化为单模态回归。
- [ ] **核心网络 (Sequence + Substrate)**
  - **Late Fusion**：将 $H_e$ 与 $H_m$ 拼接 (Concatenation) 后输入 MLP。
  - **Cross-Attention Fusion**：以底物特征为 Query，酶序列特征为 Key/Value，计算交叉注意力矩阵，模拟底物寻找活性位点的物理过程。
- [ ] **进阶网络 (Sequence + Substrate + Structure)**
  - 采用分阶段融合 (Staged Fusion)：先对齐 Sequence 与 Structure 获得结构感知表征，再与 Substrate 进行交互。

### 阶段 4：严格的训练与验证策略 (Rigorous Training & Validation)

为了确保模型不是在死记硬背数据，必须建立具备生物学意义的评估协议。

- [ ] **样本外划分 (Out-of-Distribution / OOD Split)** *【关键】*
  - 绝对禁止随机划分 (Random Split)。
  - **实现方法**：使用 `MMseqs2` 或 `CD-HIT` 根据氨基酸序列相似度（如 30% 或 50% 阈值）进行聚类划分。确保测试集中的酶序列对模型而言是“全新”的。
- [ ] **损失函数与不确定性估计 (Loss & Uncertainty)**
  - 除了标准的 MSE / MAE 回归损失外，项目建议实现**异方差高斯负对数似然 (Heteroscedastic Gaussian NLL)**。
  - 模型输出由单一标量变为均值与方差：$[\hat{y}, \hat{\sigma}^2]$。
  - 目标函数：$\min_{\phi} \mathcal{L}_{\text{pred}} = \frac{(y - \hat{y})^2}{2\hat{\sigma}^2} + \frac{1}{2}\log(\hat{\sigma}^2)$。这能让模型在预测时，同步输出其“置信度”。
- [ ] **多任务学习 (Multi-task Learning)**
  - 如果数据允许，可设计 Multi-head 输出，同时预测 $k_{\text{cat}}$ 与 $K_m$（因为 $k_{\text{cat}}/K_m$ 代表催化效率，两者存在物理耦合）。

### 阶段 5：多维度评估与消融分析 (Evaluation & Analysis)

- [ ] **核心指标计算**：
  - $R^2$ (决定系数)、Pearson Correlation Coefficient (PCC)、MAE、RMSE。
- [ ] **模型校准 (Calibration) 评估**：
  - 若使用了不确定性估计，需绘制预测误差与预测方差的散点图，证明高不确定性（High Variance）对应着高误差。
- [ ] **酶类别 (EC-Class) 细分评估**：
  - 拆解模型在 EC1 (氧化还原酶) 到 EC6 (连接酶) 上的分别表现，探讨在哪类反应中模型表现最差。
- [ ] **模态贡献度分析 (Ablation Analysis)**：
  - 量化比较“仅序列 vs 序列+底物 vs 序列+底物+结构”的 $R^2$ 增益，论证结构信息何时是必要的（例如在序列相似度极低的 OOD 测试集中）。

## 📝 三、 最终学术交付物要求 (Academic Deliverables)

1. **可复现的开源代码库 (Reproducible Codebase)**：
   - 包含清晰的 `README.md`、`requirements.txt` / `environment.yml`。
   - 数据处理、模型定义、训练脚本应模块化解耦 (Modularized)。
2. **课程设计技术报告 (Technical Report in PDF)**：
   - **Introduction**：定义问题的生化背景与机器学习挑战。
   - **Methodology**：详细叙述你采用的表征学习方法 (包含数学方程式) 以及多模态融合策略。
   - **Experimental Setup**：详细记录 OOD 划分策略、超参数设定 (Learning rate, Batch size, Epochs)、硬件环境。
   - **Results & Discussion**：
     - 表格对比各模型的评估指标。
     - Scatter plots (预测值 vs 真实值)。
     - 针对结构信息有效性的深度探讨。
3. **答辩幻灯片 (Presentation Slides)**：
   - 侧重于“你如何解决数据异质性”与“多模态架构为何有效”的逻辑推演。