# PL-Hydra-template
PyTorch Lightning + Hydra Minimal Working Example to reproduce an issue with DDP mode (see issue [here](https://forums.pytorchlightning.ai/t/using-hydra-ddp/567)) 

## Installation

Developed with Python 3.7.9

```
$ git clone git@github.com:inzouzouwetrust/PL-Hydra-template.git
$ cd PL-Hydra-template && pip install -r requirements.txt
```

## Usage

Launch a training on CPU to debug:

```
$ python train.py trainer=debug_cpu
```

Launch a training on 2 GPUs using DDP to reproduce the issue:

```
$ python train.py
```
