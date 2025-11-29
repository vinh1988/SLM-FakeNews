
# An Approach Based on Fine-Tuning Small Language Models for Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

**Authors:** [Khac-Lap Phan](mailto:lap4654100006@st.qnu.edu.vn)¹ and [Quang-Hung Le](mailto:lequanghung@qnu.edu.vn)²* **Affiliation:** Department of Information Technology, Quy Nhon University, Vietnam  
¹`lap4654100006@st.qnu.edu.vn`, ²`lequanghung@qnu.edu.vn` (Corresponding author)

---

## 📖 Abstract
The rapid proliferation of fake news on social media platforms poses significant societal challenges. While Large Language Models (LLMs) achieve high accuracy in detecting misinformation, their substantial computational costs hinder deployment in resource-constrained environments. 

This study proposes a **lightweight approach utilizing Small Language Models (SLMs)** to balance detection performance with efficiency. We implement a comparative pipeline utilizing both **Full Fine-Tuning** and **Low-Rank Adaptation (LoRA)** to evaluate three SLM architectures (**DistilBERT, MiniLM, ALBERT**) against a standard **BERT-base** baseline across three diverse benchmarks: **WELFake, LIAR, and FakeNewsNet**.

Experimental results demonstrate that SLMs maintain remarkable robustness, achieving **96–99% of the teacher model’s performance** while reducing parameter counts by up to **90%**. Notably, on the WELFake dataset, **MiniLM** achieves an F1-score of **98.33%** (within 0.4% of BERT-base) with a **3× increase in inference throughput**. Furthermore, on the highly imbalanced FakeNewsNet dataset, DistilBERT matches BERT-base’s F1-score (≈83.6%) while significantly lowering training loss.

## 🚀 Key Features
* **Small Language Models (SLMs):** Focused on efficient architectures:
    * DistilBERT
    * MiniLM
    * ALBERT
* **Fine-Tuning Strategies:**
    * Full Fine-Tuning
    * Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation).
* **Comprehensive Evaluation:** Tested on WELFake, FakeNewsNet, and LIAR datasets.
* **High Efficiency:** Achieves comparable accuracy to BERT-base with significantly lower latency and computational cost.

## 📊 Performance Highlights

| Model | Method | Dataset | F1-Score | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **BERT-base** | Baseline | WELFake | ~98.7% | 1x |
| **MiniLM** | **LoRA** | **WELFake** | **98.33%** | **3x** |
| **DistilBERT** | Full FT | FakeNewsNet| ~83.6% | High |

*(Refer to the paper for the full results table)*

## 📂 Project Structure
```text
repo/
├── data/                   # Dataset preprocessing scripts
│   ├── welfake_process.py
│   ├── liar_process.py
│   └── fakenewsnet_process.py
├── src/                    # Source code
│   ├── models.py           # SLM definitions (DistilBERT, MiniLM, ALBERT)
│   ├── train.py            # Training loop (Full FT & LoRA)
│   ├── evaluate.py         # Metrics calculation (Accuracy, F1, Latency)
│   └── utils.py            # Helper functions
├── configs/                # Hyperparameters for each model
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Dependencies
└── README.md
