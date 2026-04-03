# 🛒 Hybrid ML/AI System for E-commerce Conversion Prediction & Automated Personalization

<div align="center">

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=leaflet&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek_LLM-4B8BBE?style=for-the-badge&logo=openai&logoColor=white)

**Sorbonne Data Analytics — Master's Thesis | Academic Year 2024–2025**
**Author:** Ecem Bayındır | **Supervisor:** Marc-Arthur Diaye

</div>

---

## 🚀 Objective

This project builds and validates a **privacy-preserving, end-to-end hybrid ML/AI pipeline** for e-commerce platforms — designed especially for **small and medium-sized enterprises (SMEs)** who cannot afford costly cloud-based personalization APIs.

The system combines:
- **LightGBM** for high-accuracy conversion prediction on imbalanced behavioral data
- **SHAP** for model explainability and actionable business intelligence
- **DeepSeek-7B (local LLM)** for automated, contextually personalized marketing messages

All of this runs on **standard consumer hardware** (Apple M3 MacBook Pro, 18GB RAM) — no cloud dependencies, no external APIs, and **fully GDPR-compliant**.

---

## 📂 Project Summary

The system implements a full data science workflow on the [RetailRocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (2.76M behavioral events, 1.4M users, 0.83% conversion rate), tackling an extreme **119:1 class imbalance**:

| Stage | Description |
|---|---|
| **Data Engineering** | Load, clean, and aggregate 2.76M events across visitors and item properties |
| **Feature Engineering** | Visitor-level behavioral aggregation + category-level metadata pivoting |
| **Modeling** | LightGBM binary classifier with class imbalance handling |
| **Explainability** | SHAP global + dependence + interaction analyses |
| **Benchmarking** | LightGBM vs. Logistic Regression vs. Random Forest |
| **LLM Personalization** | DeepSeek-7B generates targeted messages for high-intent visitors |
| **Pipeline Integration** | End-to-end real-time conversion scoring + message generation |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas / NumPy** | Data manipulation and feature engineering |
| **LightGBM** | Primary classification model |
| **Scikit-Learn** | Baseline models, metrics, cross-validation |
| **SHAP** | Model interpretability and business insight extraction |
| **Matplotlib / Seaborn** | Visualizations and thesis figures |
| **DeepSeek-7B via LM Studio** | Local LLM for personalized message generation |

---

## 📊 Key Results

### ✅ Four Hypotheses Tested — Three Fully Validated

**H1 — LightGBM outperforms baselines on imbalanced data ✅**
| Model | ROC AUC | Training Time |
|---|---|---|
| **LightGBM** | **99.92%** | **1.4s** |
| Random Forest | 99.61% | ~18s |
| Logistic Regression | 90.76% | 17.9s |

> LightGBM was **12.9× faster** than Logistic Regression with the highest AUC.

**H2 — SHAP reveals interpretable business intelligence ✅**
- Top predictors: `session_duration`, `n_views`, `n_total_events`
- Strongest feature interaction: `n_views ↔ session_duration` (strength: **0.676**)
- SHAP enables actionable targeting: identify *why* a visitor is likely to convert

**H3 — Local LLM generates coherent personalized messages ⚠️ Partially Validated**
- DeepSeek-7B produces contextually relevant personalized promotional messages
- Cold-start latency: ~14 seconds (model loading); warm-run projected under 5 seconds
- Example output:
  > *"Check out your cart! With two items in category 1051, don't miss out — use code SPECIAL10 for 10% off your purchase today!"*

**H4 — Integrated hybrid pipeline is production-viable ✅**
- Full end-to-end pipeline validated on consumer hardware
- No cloud API dependencies — fully local and GDPR-compliant
- Preloading strategy reduces LLM latency to acceptable production levels

---

## 🏗️ System Architecture

```
Raw Data (RetailRocket)
        ↓
  Data Preprocessing
  (2.76M events → visitor-level features)
        ↓
  LightGBM Conversion Prediction
  (ROC AUC: 99.92%)
        ↓
  SHAP Explainability Layer
  (Feature importance + business insights)
        ↓
  High-Intent Visitor Identification
  (Ranked by predicted conversion probability)
        ↓
  DeepSeek-7B LLM (Local)
  (Personalized promotional message generation)
        ↓
  Output: Conversion Score + Personalized Message
```

---

## 📚 Dataset

**RetailRocket E-commerce Dataset**
- Source: [Kaggle — RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Events: 2,756,101 behavioral events (views, add-to-cart, transactions)
- Users: ~1.4 million unique visitors
- Conversion rate: 0.83% (11,719 converted / 1,407,580 visitors)
- Files used: `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`

> ⚠️ Dataset files are not included in this repository due to size. Please download from the Kaggle link above and place them in a `/dataset` folder.

> ✅ `ranked_visitors.csv` is a precomputed file containing visitors ranked by predicted conversion probability above a set threshold. It is used directly in Step 17 of the notebook to avoid rerunning the full scoring pipeline (which is computationally expensive) on every execution. Due to file size it is not included in this repository — feel free to reach out via [LinkedIn](https://www.linkedin.com/in/ecembayindir) or [email](mailto:ecmbyndr@gmail.com) to request it.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ecembayindir/hybrid-ml-ai-ecommerce-personalization.git
cd hybrid-ml-ai-ecommerce-personalization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) and place files in:
```
dataset/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

### 4. (Optional) Set up local LLM
- Install [LM Studio](https://lmstudio.ai/)
- Download and load `DeepSeek-7B` model locally
- Start the local inference server on `localhost`

### 5. Run the notebook
```bash
jupyter notebook Bayindir_Ecem_memoire_notebook_SDA_2024_2025.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
lightgbm
scikit-learn
shap
matplotlib
seaborn
requests
jupyter
```

---

## 🔬 Research Contributions

This thesis makes **three novel contributions**:

1. **First hybrid ML/AI architecture** for privacy-preserving e-commerce personalization deployable on SME-grade hardware
2. **SHAP-based interpretability framework** bridging technical model outputs to actionable business intelligence
3. **Reproducible local LLM deployment** eliminating cloud dependencies while maintaining full GDPR compliance

---

## ⚠️ Limitations & Future Work

| Limitation | Future Direction |
|---|---|
| Historical dataset (2015) | Validate on contemporary e-commerce data |
| LLM cold-start latency (~14s) | Optimize with persistent model preloading |
| No live A/B testing | Implement real-world A/B validation framework |
| No CRM integration | Build API bridge to CRM/CDP platforms |
| Single LLM tested | Benchmark multiple local LLMs (Llama, Mistral) |

---

## 📜 Academic Reference

> BAYINDIR, Ecem. *Hybrid Machine Learning and Artificial Intelligence System for E-commerce Conversion Prediction and Automated Personalization.* Master's Thesis, Sorbonne Data Analytics, Université Paris 1 Panthéon-Sorbonne, 2024–2025.

---

## 🤝 Connect With Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ecembayindir)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ecmbyndr@gmail.com)

</div>

<br>
<p align="center">© 2025 Ecem Bayındır. All rights reserved.</p>
<hr/>
<p align="center">
  <img src="https://komarev.com/ghpvc/?username=ecembayindir&repo=hybrid-ml-ai-ecommerce-personalization&label=Repository%20views&color=0e75b6&style=flat" alt="Repository Views">
</p>
