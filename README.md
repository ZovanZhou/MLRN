# MLRN

The source code for ''Multi-grained Label Refinement Network with Dependency Structures for Joint Intent Detection and Slot Filling''

[[Paper link]](https://arxiv.org/abs/2209.04156)

## Preparation

1. Run the commond to install the packages.

```shell
pip install -r requirements.txt
```

2. Download the [BERT](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip) pretrained models, and unzip into this folder.

## Training

1. Run the shell script to train and evaluate the model.

```shell
./train.atis.sh
```

or

```shell
./train.snips.sh
```
