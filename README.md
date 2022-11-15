# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)


This repository is the official PyTorch implementation of our AAAI-2022 [paper](https://arxiv.org/abs/2105.02446), in which we propose DiffSinger (for Singing-Voice-Synthesis) and DiffSpeech (for Text-to-Speech).
 

:tada: :tada: :tada: **Updates**:
 - Sep.11, 2022: :electric_plug: [DiffSinger-PN](docs/README-SVS-opencpop-pndm.md). Add plug-in [PNDM](https://arxiv.org/abs/2202.09778), ICLR 2022 in our laboratory, to accelerate DiffSinger freely.
 - Jul.27, 2022: Update documents for [SVS](docs/README-SVS.md). Add easy inference [A](docs/README-SVS-opencpop-cascade.md#4-inference-from-raw-inputs) & [B](docs/README-SVS-opencpop-e2e.md#4-inference-from-raw-inputs); Add Interactive SVS running on [Hugging Face🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger).
 - Mar.2, 2022: MIDI-B-version.
 - Mar.1, 2022: [NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB), for singing voice beautifying, has been released.
 - Feb.13, 2022: [NATSpeech](https://github.com/NATSpeech/NATSpeech), the improved code framework, which contains the implementations of DiffSpeech and our NeurIPS-2021 work [PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) has been released. 
 - Jan.29, 2022: support MIDI-A-version SVS.
 - Jan.13, 2022: support SVS, release PopCS dataset.
 - Dec.19, 2021: support TTS. [Hugging Face🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 
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

## Documents
- [Run DiffSpeech (TTS version)](docs/README-TTS.md).
- [Run DiffSinger (SVS version)](docs/README-SVS.md).

## Overview
| Mel Pipeline                                                                                | Dataset                                                  | Pitch Input       | F0 Prediction |   Acceleration Method       | Vocoder                       |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------| ----------------- | ------------- | --------------------------- | ----------------------------- |
| [DiffSpeech (Text->F0, Text+F0->Mel, Mel->Wav)](docs/README-TTS.md)                         | [Ljspeech](https://keithito.com/LJ-Speech-Dataset/)      | None              | Explicit      | Shallow Diffusion           | NSF-HiFiGAN                   |
| [DiffSinger (Lyric+F0->Mel, Mel->Wav)](docs/README-SVS-popcs.md)                            | [PopCS](https://github.com/MoonInTheRiver/DiffSinger)    | Ground-Truth F0   | None          | Shallow Diffusion           | NSF-HiFiGAN                   |
| [DiffSinger (Lyric+MIDI->F0, Lyric+F0->Mel, Mel->Wav)](docs/README-SVS-opencpop-cascade.md) | [OpenCpop](https://wenet.org.cn/opencpop/)               | MIDI              | Explicit      | Shallow Diffusion           | NSF-HiFiGAN                   |
| [FFT-Singer (Lyric+MIDI->F0, Lyric+F0->Mel, Mel->Wav)](docs/README-SVS-opencpop-cascade.md) | [OpenCpop](https://wenet.org.cn/opencpop/)               | MIDI              | Explicit      | Invalid                     | NSF-HiFiGAN                   |
| [DiffSinger (Lyric+MIDI->Mel, Mel->Wav)](docs/README-SVS-opencpop-e2e.md)                   | [OpenCpop](https://wenet.org.cn/opencpop/)               | MIDI              | Implicit      | None                        | Pitch-Extractor + NSF-HiFiGAN |
| [DiffSinger+PNDM (Lyric+MIDI->Mel, Mel->Wav)](docs/README-SVS-opencpop-pndm.md)             | [OpenCpop](https://wenet.org.cn/opencpop/)               | MIDI              | Implicit      | PLMS                        | Pitch-Extractor + NSF-HiFiGAN |
 

## Tensorboard
```sh
tensorboard --logdir_spec exp_name
```
<table style="width:100%">
  <tr>
    <td><img src="resources/tfb.png" alt="Tensorboard" height="250"></td>
  </tr>
</table>

## Audio Demos
Old audio samples can be found in our [demo page](https://diffsinger.github.io/). Audio samples generated by this repository are listed here:

### TTS audio samples
Speech samples (test set of LJSpeech) can be found in [resources/demos_1213](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_1213). 

### SVS audio samples
Singing samples (test set of PopCS) can be found in [resources/demos_0112](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0112).

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
