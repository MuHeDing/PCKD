<div align="center">

<h2 align="center">
<b>PCKD: Preview-based Category Contrastive Learning for Knowledge Distillation</b>
</h2>

<div>
Muhe Ding, Jianlong Wu, Xue Dong, Xiaojie Li, Pengda Qin, Tian Gan, Liqiang Nie
</div>

<br>

<div align="center">
    <a href="https://ieeexplore.ieee.org/document/10880570" target="_blank">
    <img src="https://img.shields.io/badge/Journal-TCSVT%202025-blue" alt="Journal"></a>
    <a href="https://arxiv.org/abs/2410.14143" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper arXiv"></a>
</div>

</div>

---

## 🔥 Overview

This paper has been **accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT 2025)**.

We propose a novel knowledge distillation framework that enhances student learning by modeling **category-level structures** rather than only instance-level alignment.

Our method introduces:

- **Category Contrastive Learning** to capture inter-class relationships  
- **Preview Strategy** to adaptively adjust learning based on sample difficulty  

This leads to more **discriminative representations** and improved generalization.

---

## 🚀 Method

<img src="./assets/framework2.png">

Given an input, the teacher provides a **preview signal**, which guides the student to learn category-aware representations.

The framework consists of:

- **Feature Alignment**
- **Category Contrastive Learning**
- **Preview-guided Optimization**

---

## 🧩 Key Components

### 🔹 Category Contrastive Learning
- Align instance features with category centers  
- Improve intra-class compactness and inter-class separability  

---

### 🔹 Preview Strategy
- Dynamically estimate sample difficulty  
- Assign adaptive learning weights  
- Improve training stability  

---

### 🔹 Unified Objective
- Knowledge Distillation Loss  
- Contrastive Loss  
- Preview-guided Weighting  

---

## 📊 Results

<img src="./assets/cifar-10.png">

- Outperforms state-of-the-art KD methods  
- Benchmarks:
  - CIFAR-100  
  - ImageNet  

---

## 🧪 Usage

### Installation

```bash
git clone https://github.com/yourname/PCKD.git
cd PCKD
pip install -r requirements.txt
