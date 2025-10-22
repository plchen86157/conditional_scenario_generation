# Controllable Collision Scenario Generation via Collision Pattern Prediction

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge)](https://plchen86157.github.io/conditional_scenario_generation/)
[![Video Overview](https://img.shields.io/badge/Video-Overview-red?style=for-the-badge)](https://www.youtube.com/watch?v=W-_sarZqfMo)
[![arXiv](https://img.shields.io/badge/arXiv-2510.12206-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2510.12206)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](LICENSE)

---

This repository contains the official code for **Controllable Collision Scenario Generation via Collision Pattern Prediction**, a method for controllable collision scenario generation in autonomous driving.

**Authors:** Pin-Lun Chen, Chi-Hsi Kung, Che-Han Chang, Wei-Chen Chiu, Yi-Ting Chen  
**Affiliation:** [National Yang Ming Chiao Tung University](https://www.nycu.edu.tw)

---
<!-- 
### 🔗 Quick Links
- 🧠 [**Paper Preprint (arXiv 2510.12206)**](https://arxiv.org/abs/2510.12206)  
- 🎥 [**Video Overview (YouTube)**](https://www.youtube.com/watch?v=W-_sarZqfMo)  
- 💻 [**Project Page / Code Repository**](https://github.com/plchen86157/conditional_scenario_generation)

--- -->

<p align="center">
  <img src="images/Pattern crashes.gif" width="85%">
</p>

*We introduce Collision Pattern, a compact and interpretable representation of the relative configuration between ego-attacker at the collision moment. Given a safe scenario, the user specifies collision type and time-to-accident (TTA) to predict collision pattern. This pattern guides the quintic motion planner to generate a feasible attacker trajectory that realizes the specified collision.*


### System Requirements
* Linux ( Tested on Ubuntu 18.04 )
* Python3 ( Tested on Python 3.8 )
* PyTorch ( Tested on PyTorch 1.8.0 )
* CUDA ( Tested on CUDA 11.1 )
* GPU ( Tested on Nvidia RTX3090Ti )
* CPU ( Tested on Intel Core i7-12700, 12-Core 20-Thread )

* [NuScenes-api](https://www.nuscenes.org/nuscenes#download)

<!-- <p align="center">
  <img src="images/model.png" width="85%">
</p> -->

## Usage

### Preprocessing

To preprocess our COLLIDE data into vectorized representation:
```
bash scripts/preprocessing_data.bash
```

### Training

To train the condition collision scenario generation model with COLLIDE:
```
bash scripts/train.bash
```

### Inference

To generate prediction results:
```
python test.py
```

### Video visualization

To generate video result based on .csv files created in the inference stage:
```
python plot_and_metric.py
```

### Citation

```
@article{chen2025controllable,
  title={Controllable Collision Scenario Generation via Collision Pattern Prediction},
  author={Chen, Pin-Lun and Kung, Chi-Hsi and Chang, Che-Han and Chiu, Wei-Chen and Chen, Yi-Ting},
  journal={arXiv preprint arXiv:2510.12206},
  year={2025}
}
```