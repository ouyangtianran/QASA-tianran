# QASA

Official code for the ECCV 2026 paper:

> **QASA: Quality-Guided K-Adaptive Slot Attention for Unsupervised Object-Centric Learning**

## Installation

```bash
conda env create -f environment.yml
conda activate qasa
```

## Datasets

Please follow the [SPOT repository](https://github.com/gkakogeorgiou/spot) for
dataset downloads and directory structures.

## Training

```bash
DATA_PATH=/path/to/COCO2017 bash scripts/qasa_train_ddp_coco.sh
DATA_PATH=/path/to/COCO2017 bash scripts/qasa_train_ddp_mlp_coco.sh

DATA_PATH=/path/to/MOVi/c bash scripts/qasa_train_ddp_movi.sh
DATA_PATH=/path/to/MOVi/c bash scripts/qasa_train_ddp_mlp_movi.sh

DATA_PATH=/path/to/VOCdevkit/VOC2012 bash scripts/qasa_train_ddp_voc.sh
DATA_PATH=/path/to/VOCdevkit/VOC2012 bash scripts/qasa_train_ddp_mlp_voc.sh
```

## Evaluation

```bash
DATA_PATH=/path/to/COCO2017 \
CHECKPOINT_PATH=/path/to/checkpoint.pt.tar \
bash scripts/qasa_eval_coco.sh

DATA_PATH=/path/to/MOVi/c \
CHECKPOINT_PATH=/path/to/checkpoint.pt.tar \
bash scripts/qasa_eval_movi.sh

DATA_PATH=/path/to/VOCdevkit/VOC2012 \
CHECKPOINT_PATH=/path/to/checkpoint.pt.tar \
bash scripts/qasa_eval_voc.sh
```

## Acknowledgements

This codebase is built upon [SPOT](https://github.com/gkakogeorgiou/spot). We
thank the authors for releasing their code.
