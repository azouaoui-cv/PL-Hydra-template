# PL-Hydra-template
PyTorch Lightning + Hydra template to use DDP

## Installation

Developed with Python 3.7.9

```
$ git clone git@github.com:inzouzouwetrust/PL-Hydra-template.git
$ cd PL-Hydra-template && pip install -r requirements.txt
```

**Warning**

A `data` folder will be created when using the `train.py` script, holding the `CIFAR10` data, so make sure you have enough storage available.

## Usage

Launch a training on CPU to debug:

```
$ python train.py trainer=debug_cpu
```

Launch a training on 2 GPUs using DDP:

```
$ python train.py
```
