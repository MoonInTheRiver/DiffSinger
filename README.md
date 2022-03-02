# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 | [中文文档](README-zh.md)

This repository is the official PyTorch implementation of our AAAI-2022 [paper](https://arxiv.org/abs/2105.02446), in which we propose DiffSinger (for Singing-Voice-Synthesis) and DiffSpeech (for Text-to-Speech).
 
<table style="width:100%">
  <tr>
    <th>DiffSinger/DiffSpeech at training</th>
    <th>DiffSinger/DiffSpeech at inference</th>
  </tr>
  <tr>
    <td><img src="resources/model_a.png" alt="Training" height="300"></td>
    <td><img src="resources/model_b.png" alt="Inference" height="300"></td>
  </tr>
</table>

:tada: :tada: :tada: **Updates**:
 - Mar.2, 2022: [MIDI-new-version](usr/configs/midi/readme-e2e.md): A substantial improvement :sparkles:
 - Mar.1, 2022: [NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB), for singing voice beautifying, has been released  :sparkles:  :sparkles:  :sparkles: .
 - Feb.13, 2022: [NATSpeech](https://github.com/NATSpeech/NATSpeech), the improved code framework, which contains the implementations of DiffSpeech and our NeurIPS-2021 work [PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) has been released :sparkles: :sparkles: :sparkles:. 
 - Jan.29, 2022: support [MIDI-old-version](usr/configs/midi/readme.md) SVS. **Keep Updating**. :construction: :pick: :hammer_and_wrench: 
 - Jan.13, 2022: support SVS, release PopCS dataset.
 - Dec.19, 2021: support TTS. [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 
:rocket: **News**: 
 - Feb.24, 2022: Our new work, NeuralSVB was accepted by ACL-2022 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277). [Demo Page](https://neuralsvb.github.io).
 - Dec.01, 2021: DiffSinger was accepted by AAAI-2022.
 - Sep.29, 2021: Our recent work `PortaSpeech: Portable and High-Quality Generative Text-to-Speech` was accepted by NeurIPS-2021 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2109.15166) .
 - May.06, 2021: We submitted DiffSinger to Arxiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446).

## Environments
```sh
conda create -n your_env_name python=3.8
source activate your_env_name 
pip install -r requirements_2080.txt   (GPU 2080Ti, CUDA 10.2)
or pip install -r requirements_3090.txt   (GPU 3090, CUDA 11.4)
```

## DiffSpeech (TTS version)
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

First, you need a pre-trained FastSpeech2 checkpoint. You can use the [pre-trained model](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip), or train FastSpeech2 from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config configs/tts/lj/fs2.yaml --exp_name fs2_lj_1 --reset
```
Then, to train DiffSpeech, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/lj_ds_beta6.yaml` to fit your path.

### 3. Inference Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_ds_beta6_1213 --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/lj_ds_beta6_1213.zip);
 - the individual pre-trained model of [FastSpeech 2](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/fs2_lj_1.zip) for the shallow diffusion mechanism in DiffSpeech;
 
Remember to put the pre-trained models in `checkpoints` directory.

## DiffSinger (SVS version)

### 0. Data Acquirement
- See in [apply_form](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md).
- Dataset [preview](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_preview.zip).

### 1. Preparation
#### Data Preparation
a) Download and extract PopCS, then create a link to the dataset folder: `ln -s /xxx/popcs/ data/processed/popcs`

b) Run the following scripts to pack the dataset for training/inference.
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/popcs_ds_beta6.yaml
# `data/binary/popcs-pmf0` will be generated.
```

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) which is specially designed for SVS with NSF mechanism.
Please unzip this file into `checkpoints` before training your acoustic model.

(Update: You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory)

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the [pre-trained model](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip), or train FFT-Singer from scratch, run:

```sh
# First, train fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset
# Then, infer fft-singer;
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_fs2.yaml --exp_name popcs_fs2_pmf0_1230 --reset --infer 
```

Then, to train DiffSinger, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/popcs_ds_beta6_offline.yaml` to fit your path.

### 3. Inference Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_ds_beta6_offline_pmf0_1230 --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_ds_beta6_offline_pmf0_1230.zip);
 - the pre-trained model of [FFT-Singer](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/popcs_fs2_pmf0_1230.zip) for the shallow diffusion mechanism in DiffSinger;

Remember to put the pre-trained models in `checkpoints` directory.

*Note that:* 

- *the original PWG version vocoder in the paper we used has been put into commercial use, so we provide this HifiGAN version vocoder as a substitute.*
- *we assume the ground-truth F0 to be given as the pitch information following [1][2][3]. If you want to conduct experiments on MIDI data, you need an external F0 predictor (like [MIDI version of DiffSinger](usr/configs/midi/readme.md)) or a joint prediction with spectrograms.*

[1] Adversarially trained multi-singer sequence-to-sequence singing synthesizer. Interspeech 2020.

[2] SEQUENCE-TO-SEQUENCE SINGING SYNTHESIS USING THE FEED-FORWARD TRANSFORMER. ICASSP 2020.

[3] DeepSinger : Singing Voice Synthesis with Data Mined From the Web. KDD 2020.

## Tensorboard
```sh
tensorboard --logdir_spec exp_name
```
<table style="width:100%">
  <tr>
    <td><img src="resources/tfb.png" alt="Tensorboard" height="250"></td>
  </tr>
</table>

## Mel Visualization
Along vertical axis, DiffSpeech: [0-80]; FastSpeech2: [80-160].

<table style="width:100%">
  <tr>
    <th>DiffSpeech vs. FastSpeech 2</th>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2-1.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
  <tr>
    <td><img src="resources/diffspeech-fs2-2.png" alt="DiffSpeech-vs-FastSpeech2" height="250"></td>
  </tr>
</table>

## Audio Demos
Audio samples can be found in our [demo page](https://diffsinger.github.io/).

We also put part of the audio samples generated by DiffSpeech+HifiGAN (marked as [P]) and GTmel+HifiGAN (marked as [G]) of test set in [resources/demos_1213](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_1213). 

(corresponding to the pre-trained model [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/lj_ds_beta6_1213.zip))

---
:rocket: :rocket: :rocket: **Update:**

New singing samples can be found in [resources/demos_0112](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0112).

## Citation
    @article{liu2021diffsinger,
      title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
      author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
      journal={arXiv preprint arXiv:2105.02446},
      volume={2},
      year={2021}}


## Acknowledgements
Our codes are based on the following repos:
* [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
* [DiffWave](https://github.com/lmnt-com/diffwave)

Also thanks [Keon Lee](https://github.com/keonlee9420/DiffSinger) for fast implementation of our work.
