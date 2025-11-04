# 🤖 Multi-Source Cross-Project Software Defect Prediction (DistilBERT + GRU Hybrid)

## 📘 Overview
This repository presents a **hybrid deep learning framework** for **software defect prediction** that integrates **DistilBERT** (a lightweight transformer-based language model) with a **bidirectional GRU**.  
The model bridges **transfer learning and feature sequence modeling**, enabling **cross-project defect prediction** from heterogeneous datasets.

The approach converts numerical software metrics into **semantic text embeddings**, allowing DistilBERT to extract contextual representations before refinement through a GRU network. This architecture captures both **feature-level dependencies** and **temporal patterns** across multiple projects.

---

## 🧠 Methodology

### 1. **Data Preparation**
- Dataset used: `jm1.csv` (NASA PROMISE defect dataset).
- All missing values handled and normalized.
- The **target column (`defects`)** is binarized (`1 = defective`, `0 = clean`).
- Class imbalance mitigated using **SMOTE** (Synthetic Minority Oversampling Technique).
- PCA visualization used to confirm effective balancing and feature spread.

### 2. **Feature-to-Text Transformation**
Each sample’s numerical features are converted into text-like representations, e.g.  
> `"feature values: 0.532 1.014 3.521 ..."`

This lets DistilBERT interpret structured numeric data as natural-language-style tokens — enabling **transfer learning from pre-trained NLP models** on software metric data.

### 3. **Hybrid Model Architecture**

| Component | Description |
|------------|-------------|
| **DistilBERT** | Pretrained transformer extracts contextual embeddings from textualized features. |
| **Bidirectional GRU** | Captures sequential correlations and temporal dependencies. |
| **Classifier Head** | Linear + Dropout layer for binary defect prediction. |
| **Optimizer** | AdamW with layer-wise learning rates. |
| **Scheduler** | Linear warmup + decay using Hugging Face’s scheduler. |

**Hybrid Forward Pass:**
Numeric features → Text representation → DistilBERT embeddings
→ Bi-GRU feature refinement → Dense classifier → Prediction


---

## 🧩 Architecture Summary

| Component | Details |
|------------|----------|
| Base Model | DistilBERT (Transformer Encoder) |
| RNN Layer | Bi-GRU (2 layers, hidden size = 256) |
| Dropout | 0.3 |
| Optimizer | AdamW |
| LR Scheduler | Linear warmup/decay |
| Loss Function | CrossEntropyLoss |
| Training Epochs | Up to 10 with early stopping |
| Device | CUDA / CPU supported |

---


