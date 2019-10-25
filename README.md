# Severstal: Steel Defect Detection

This is an another one approach to solve the competition from kaggle
[Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection).

21th place out of 2435 (silver medal) with 0.90517 dice score (top 1 -- 0.90883).
Actually this competition was with poor chosen metric. The ground truth masks has inaccurate labels
so, if we move our predictions on one pixel, the metric will change in third decimal places.
So this competition is completely lottery.

### Prerequisites

- GPU with >27 Gb RAM (e.g. Tesla V100)
- [NVidia apex](https://github.com/NVIDIA/apex) (otherwise)

```bash
pip install -r requirements.txt
```

### Usage

First download the train and test data from the competition link.

To train the model run

```bash
bash ./run.sh
```

This will generates trained models.
