# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)

This repository is the official PyTorch implementation of our AAAI-2022 [paper](https://arxiv.org/abs/2105.02446), in which we propose DiffSinger (for Singing-Voice-Synthesis) and DiffSpeech (for Text-to-Speech).
 
Besides, more detailed & improved code framework, which contains the implementations of FastSpeech 2, DiffSpeech and our NeurIPS-2021 work [PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) is coming soon :sparkles: :sparkles: :sparkles:.
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

:rocket: **News**: 
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
### 1. Data Preparation

a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then create a link to the dataset folder: `ln -s /xxx/LJSpeech-1.1/ data/raw/`

b) Download and Unzip the [ground-truth duration](https://drive.google.com/file/d/1SqwIISwaBZDiCW1MHTHx-MKX6_NQJ_f4/view?usp=sharing) extracted by [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz):  `tar -xvf mfa_outputs.tar; mv mfa_outputs data/processed/ljspeech/`

c) Run the following scripts to pack the dataset for training/inference.

```sh
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml

# `data/binary/ljspeech` will be generated.
```

### 2. Training Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_exp1 --reset
```


### 3. Inference Example

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/lj_ds_beta6.yaml --exp_name lj_exp1 --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSpeech](https://drive.google.com/file/d/1AHRuNS379v2_lNuz4-Mjlpii7TZsfs3f/view?usp=sharing);
 - the pre-trained model of [HifiGAN](https://drive.google.com/file/d/1Z3DJ9fvvzIci9DAf8jwchQs-Ulgpx6l8/view?usp=sharing) vocoder;
 - the individual pre-trained model of [FastSpeech 2](https://drive.google.com/file/d/1Zp45YjKkkv5vQSA7woHIqEggfyLqQdqs/view?usp=sharing) for the shallow diffusion mechanism in DiffSpeech;
 
Remember to put the pre-trained models in `checkpoints` directory.

About the determination of 'k' in shallow diffusion: We recommend the trick introduced in Appendix B. We have already provided the proper 'k' for Ljspeech dataset in the config files.


## DiffSinger (SVS version)

### 0. Data Acquirement
- [ ] WIP.
We will provide a form to apply for PopCS dataset.

### 1. Data Preparation
- [ ] WIP.
Similar to DiffSpeech. 

### 2. Training Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6.yaml --exp_name popcs_exp1 --reset
# or
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_exp2 --reset
```
### 3. Inference Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/popcs_ds_beta6_offline.yaml --exp_name popcs_exp2 --reset --infer
```
The pre-trained model for SVS will be provided recently. 
<!--
Besides, the original PWG-based vocoder for SVS in our paper has been used commercially, but we are working on training a better HifiGAN-based vocoder.
-->

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

We also put part of the audio samples generated by DiffSpeech+HifiGAN (marked as [P]) and GTmel+HifiGAN (marked as [G]) of test set in resources/demos_1218. 

(corresponding to the pre-trained model [DiffSpeech](https://drive.google.com/file/d/1AHRuNS379v2_lNuz4-Mjlpii7TZsfs3f/view?usp=sharing))

## Citation
    @misc{liu2021diffsinger,
      title={DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism}, 
      author={Jinglin Liu and Chengxi Li and Yi Ren and Feiyang Chen and Zhou Zhao},
      year={2021},
      eprint={2105.02446},
      archivePrefix={arXiv},}


## Acknowledgements
Our codes are based on the following repos:
* [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)

Also thanks [Keon Lee](https://github.com/keonlee9420/DiffSinger) for fast implementation of our work.
