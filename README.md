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

While mainstream Knowledge Distillation (KD) methods successfully transfer knowledge by aligning instance-level feature representations, they often neglect **category-level information** and the inherent **difficulty of individual samples**. To address these issues, we propose **PCKD**, a novel Preview-based Category Contrastive Learning method. 

Our framework enhances student learning through two core innovations:

- **Category Contrastive Learning for Knowledge Distillation:** It distills structural knowledge by modeling both instance-level feature correspondence and the relationships between instance features and category centers. This explicit optimization yields highly discriminative category centers and better classification accuracy.
- **Preview-based Learning Strategy:** Unlike existing methods that treat all samples equally or curriculum learning that simply drops hard samples, PCKD dynamically determines learning weights based on sample difficulty. It assigns a smaller weight to hard instances—acting as a "preview"—to gently and effectively guide the student's training.

Extensive experiments demonstrate that PCKD achieves state-of-the-art performance across several challenging datasets, including **CIFAR-100, ImageNet, and Pascal VOC**.

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
  - Pascal VOC

---

## 🧪 Usage

### Installation

```bash
git clone [https://github.com/yourname/PCKD.git](https://github.com/yourname/PCKD.git)
cd PCKD
pip install -r requirements.txt
