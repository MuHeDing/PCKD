<div align="center">
<h2 align="center">
    <b>Preview-Based Category Contrastive Learning for Knowledge Distillation</b>
</h2>

<div>
Muhe Ding<sup>1</sup>,
Jianlong Wu<sup>1</sup>,
Xue Dong<sup>2</sup>,
Xiaojie Li<sup>1</sup>,
Pengda Qin<sup>3</sup>,
Tian Gan<sup>4</sup>,
Liqiang Nie<sup>1</sup>
</div>
<br>
<sup>1</sup>Department of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), China<br>
<sup>2</sup>School of Software, Tsinghua University, China<br>
<sup>3</sup>Alibaba Group, China<br>
<sup>4</sup>School of Computer Science and Technology, Shandong University, China<br>
</div>

<div align="center">
    <a href="https://ieeexplore.ieee.org/document/10880570" target="_blank">
    <img src="https://img.shields.io/badge/Paper-TCSVT%202025-blue" alt="IEEE TCSVT"></a>
</div>

## :bulb: Overview

Knowledge distillation transfers knowledge from a large teacher network to a compact student network. Existing methods usually align instance-level logits or features, but they often ignore two useful signals:

- category-level information from the classifier weights, which can be viewed as category centers;
- sample difficulty, because hard samples should not be treated exactly like easy samples at the beginning of training.

PCKD addresses these issues with two components:

- **Category Contrastive Learning for Knowledge Distillation (CKD)**: distills feature representations, category centers, and their correlations. In code, the core category contrastive loss is implemented as `CategoryConLoss` in `helper/losses.py`.
- **Preview-based Learning Strategy**: dynamically weights each sample according to its difficulty. Hard samples receive smaller weights early in training and gradually contribute more as training proceeds. The weighting logic is implemented in the `pckd` branch of `helper/loops.py`.

![PCKD framework](assets/framework2.png)

The paper evaluates PCKD on CIFAR-100, ImageNet, STL-10, and TinyImageNet. This code release focuses on the CIFAR-100 training pipeline and related KD baselines inherited from the RepDistiller/CRD codebase.


## :open_file_folder: Data and Teacher Checkpoints

Before training a student, prepare:

1. CIFAR-100 in Python format. The expected raw folder name is `cifar-100-python`.
2. A pretrained teacher checkpoint. The student script expects `--path_t` to point to a checkpoint whose dictionary contains the key `model`.

The current dataloader uses the dataset root configured in `dataset/cifar100.py`. Adjust `get_data_folder()` there to your local dataset path before running if needed.

Teacher checkpoints are commonly placed under:

```text
save/models/<teacher_run_name>/<teacher_model>_best.pth
```

For example:

```text
save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth
```


### :computer: WRN-40-2 Teacher to WRN-16-2 Student

```bash
python train_student.py \
  --path_t teacher_pth \
  --distill pckd \
  -r 1 -a 1 -b 0.05 -f 20 \
  --model_s wrn_16_2 \
  --dataset cifar100
```

## :bar_chart: Results Reported in the Paper

PCKD improves over KD and strong distillation baselines on CIFAR-100 for both same-family and heterogeneous teacher/student pairs. The paper also reports:

![CIFAR-100 results](assets/cifar-100.png)

## :hugs: Citation

If you find this repository useful, please cite:

```bibtex
@article{ding2025pckd,
  title={Preview-based Category Contrastive Learning for Knowledge Distillation},
  author={Ding, Muhe and Wu, Jianlong and Dong, Xue and Li, Xiaojie and Qin, Pengda and Gan, Tian and Nie, Liqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

## :pray: Acknowledgement

This work is built on the following repositories. Thanks to their great work.

- [RepDistiller](https://github.com/HobbitLong/RepDistiller/tree/master), for [Contrastive Representation Distillation](http://arxiv.org/abs/1910.10699) (ICLR 2020).
- [HSAKD](https://github.com/winycg/HSAKD), for [Hierarchical Self-supervised Augmented Knowledge Distillation](https://www.ijcai.org/proceedings/2021/0168.pdf) (IJCAI 2021)
- [ICKD](https://github.com/ADLab-AutoDrive/ICKD), for [Exploring Inter-Channel Correlation for Diversity-preserved Knowledge Distillation](https://arxiv.org/abs/2202.03680).
