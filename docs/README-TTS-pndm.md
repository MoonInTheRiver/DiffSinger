# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=TTSDemo)](https://huggingface.co/spaces/NATSpeech/DiffSpeech) 

## DiffSpeech (TTS)
### 1. Preparation

#### Data Preparation
a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then create a link to the dataset folder: `ln -s /xxx/LJSpeech-1.1/ data/raw/`

b) Download and Unzip the [ground-truth duration](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/mfa_outputs.tar) extracted by [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz):  `tar -xvf mfa_outputs.tar; mv mfa_outputs data/processed/ljspeech/`

c) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml

# `data/binary/ljspeech` will be generated.
```

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip) vocoder.
Please unzip this file into `checkpoints` before training your acoustic model.

### 2. Training Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_pndm.yaml --exp_name ds_pndm_lj_1 --reset
```

### 3. Inference Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_pndm.yaml --exp_name ds_pndm_lj_1 --reset --infer
```
