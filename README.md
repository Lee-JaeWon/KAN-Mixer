# KAN-Mixer
KAN-Mixer from MLP-Mixer and KAN(Kolmogorov-Arnold Networks)

## Overview
With the recent emergence of KAN, this repository replaces the MLPs in the original MLP-Mixer with Kolmogorov-Arnold Networks (KAN). Currently, only the S/16 model is implemented, but we plan to add other sizes of model and additional details soon.<br>
Training loss converges better than vanilla models.

<p align="center"><img src="/doc/loss.png" width = "600" ></p>

## Setup
```
git clone https://github.com/Lee-JaeWon/KAN-Mixer.git
conda env create --file environment.yml
conda activate kanmixer
```

## Train
```
python train_cifar100_kanmixer.py
```

## Acknowledgement
We thank original code from [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan) for presenting such an excellent work.