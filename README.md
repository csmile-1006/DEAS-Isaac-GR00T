<div align="left">
  <h1> <a href=https://arxiv.org/abs/2510.07730>DEAS</a> + Isaac-GR00T + RoboCasa </h1>
  <p style="font-size: 1.2em;">
    <a href="https://changyeon.site/deas"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/datasets/changyeon/deas_robocasa"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2510.07730"><strong>Paper</strong></a>
  </p>
  
  This repository provides the re-implementation of DEAS with Isaac GR00T N1.5 introduced in: <br/>
  <a href=https://arxiv.org/abs/2510.07730>DEAS: DEtached value learning with Action Sequence for Scalable Offline RL</a><br/>
  <a href=https://changyeon.site>Changyeon Kim</a>, Haeone Lee, <a href=https://younggyo.me>Younggyo Seo</a>, <a href=https://sites.google.com/view/kiminlee>Kimin Lee</a>, <a href=https://yukezhu.me>Yuke Zhu</a><br/>
</div>

## Installation Guide

The easiest way to set up is through the Anaconda package management system. Follow the instructions below to install all three required repositories, their dependencies, and download the assets needed for the simulation task:

```sh
# 1. Set up conda environment
conda create -n gr00t python=3.10
conda activate gr00t

# 2. Clone and install Isaac-GR00T-DEAS
git clone https://github.com/csmile-1006/Isaac-GR00T-DEAS.git
cd Isaac-GR00T-DEAS
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
cd ..

# 3. Clone and install robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
pip install -e robosuite

# 4. Clone and install robocasa-gr1-tabletop-tasks
git clone https://github.com/robocasa/robocasa.git
pip install -e robocasa

# 5. Download assets
cd robocasa
python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
```

## Getting started with this repo

### 1. Download dataset
Download the [dataset](https://huggingface.co/datasets/changyeon/deas_robocasa) using HuggingFace cli.
```bash
cd ~
hf download changyeon/deas_robocasa --repo-type dataset
```

### 2. Fine-Tuning GR00T N1.5
```bash
bash bash_scripts/finetune_gr00t.sh 30000 ${NUM_GPUS} ${BATCH_SIZE}
```

### 3. Training DEAS Critic
```bash
bash ./bash_scripts/train_deas_critic.sh 30000 16 0.9 0.99 0.7 ${NUM_GPUS} ${BATCH_SIZE}
```

### 4. Evaluation
set `CKPT_PATH` to be the path of the GR00T N1.5 trained in Section 2.\
set `CRITIC_CKPT_PATH` to be the path of the critic trained in Section 3.

```bash
# Evaluating GR00T N1.5 trained with filtered BC
bash ./bash_scripts/eval_gr00t.sh 0 ${CKPT_PATH} ${SEED} 5 50

# Evaluating GR00T N1.5 using DEAS Critic
bash ./bash_scripts/eval_deas.sh 0 ${CKPT_PATH} ${CRITIC_CKPT_PATH} ${SEED} 5 50 10 0.0
```


## Acknowledgement
This code is mainly built upon [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)  and [RoboCasa](https://github.com/robocasa/robocasa) repositories.

## Citation
```bibtex
@article{kim2025deas,
    title={DEAS: DEtached value learning with Action Sequence for Scalable Offline RL},
    author={Changyeon Kim and Haewon Lee and Younggyo Seo and Kimin Lee and Yuke Zhu},
    journal={arXiv:2510.07730},
    year={2025},
}
```
