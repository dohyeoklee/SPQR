# SPQR
author's implementation for NeurIPS 23 paper 'SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning'

## How to run
```
cd SPQR-CQL-Min
python scripts/spqr_run.py --env kitchen-mixed-v0 --policy_lr 1e-4 --lagrange_thresh 5.0 --min_q_weight 5.0 --gpu 0 --min_q_version 3 --ensemble_size 6 --beta 2.0 --seed 10
```

## Environment Setting(for our experiments)
-download anaconda <br/>
-conda create -n sqpr python=3.8 <br/>
-torch,torchvision.torchaudio: download whl in https://download.pytorch.org/whl/torch_stable.html <br/>
-python -m pip install [file.whl] <br/>
-download mujoco210 <br/>
-pip install d4rl <br/>

## Requirements(for our experiments)
ubuntu==20.04 <br/> 
python==3.8 <br/>
cuda==11.5 <br/>
pytorch==1.11.0 <br/>
torchaudio==0.11.0 <br/>
torchvision==0.12.0 <br/>
gym==0.23.1 <br/>
mujoco==2.1 <br/> 
dm_control==1.0.9 <br/>
gttimer,matplotlib,python-dateutil,tensorboardX,opencv-python <br/>
