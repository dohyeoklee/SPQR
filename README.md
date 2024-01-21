# SPQR
author's implementation for NeurIPS 23 paper 'SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning'

# How to run
```
cd SPQR-CQL-Min
python scripts/spqr_run.py --env kitchen-mixed-v0 --policy_lr 1e-4 --lagrange_thresh 5.0 --min_q_weight 5.0 --gpu 0 --min_q_version 3 --ensemble_size 6 --beta 2.0 --seed 10
```

# Environment Setting
-download anaconda
-conda create -n sqpr python=3.8
-torch,torchvision.torchaudio: download whl in https://download.pytorch.org/whl/torch_stable.html
-python -m pip install [file.whl]
-download mujoco210
-pip install d4rl

# Requirements
ubuntu==20.04
python==3.8
cuda==11.5
pytorch==1.11.0
torchaudio==0.11.0
torchvision==0.12.0
gym==0.23.1
mujoco==2.1
dm_control==1.0.9
gttimer,matplotlib,python-dateutil,tensorboardX,opencv-python
