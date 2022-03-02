# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)

Substantial update: We 1) **abandon** the extraction and explicit prediction of the F0 curve; 2) increase the receptive field of the denoiser; 3) make the linguistic encoder more robust.
**By doing so, 1) the synthesized recordings are more natural in terms of pitch; 2) the pipeline is more simpler.**

简而言之，把F0曲线的动态性交给生成式模型去捕捉，而不再是以前那样用MSE约束对数域F0。

## DiffSinger (MIDI version SVS)
- First, we tend to remind you that MIDI version is not included in the content of our AAAI paper. The camera-ready version of the paper won't be changed. Thus, the authors make no warranties regarding this part of codes/experiments.
- Second, there are many differences of model structure, especially in the **melody frontend**. 
- Third, thanks [Opencpop team](https://wenet.org.cn/opencpop/) for releasing their SVS dataset with MIDI label, **Jan.20, 2022**. (Also thanks to my co-author [Yi Ren](https://github.com/RayeRen), who applied for the dataset and did some preprocessing works for this part)

### 0. Data Acquirement
a) For PopCS dataset: WIP. We may release the MIDI label of PopCS in the future, and update this part. 

b) For Opencpop dataset: Please strictly follow the instructions of [Opencpop](https://wenet.org.cn/opencpop/). We have no right to give you the access to Opencpop.

The pipeline below is designed for Opencpop dataset:

### 1. Preparation

#### Data Preparation
Download and extract Opencpop, then create a link to the dataset folder: `ln -s /xxx/opencpop data/raw/`

#### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) which is specially designed for SVS with NSF mechanism.
Please unzip this file into `checkpoints` before training your acoustic model.

(Update: You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory)

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

#### Exp Name Preparation
```bash
export MY_DS_EXP_NAME=0228_opencpop_ds100_rel
```

```
.
|--data
    |--raw
        |--opencpop
            |--segments
                |--transcriptions.txt
                |--wavs
|--checkpoints
    |--MY_DS_EXP_NAME (optional)
    |--0109_hifigan_bigpopcs_hop128
        |--model_ckpt_steps_1512000.ckpt
        |--config.yaml
```

### 2. Training Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name MY_DS_EXP_NAME --reset  
```

### 3. Inference Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name MY_DS_EXP_NAME --reset --infer
```

We also provide:
 - the pre-trained model of DiffSinger;
 
They can be found in [here](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0228_opencpop_ds100_rel.zip).

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Some issues.
a) the HifiGAN-Singing is trained on our [vocoder dataset](https://dl.acm.org/doi/abs/10.1145/3474085.3475437) and the training set of [PopCS](https://arxiv.org/abs/2105.02446). Opencpop is the out-of-domain dataset (unseen speaker). This may cause the deterioration of audio quality, and we are considering fine-tuning this vocoder on the training set of Opencpop.

b) in this version of codes, we used the melody frontend ([lyric + MIDI]->[ph_dur]) to predict phoneme duration. F0 curve is implicitly predicted together with mel-spectrogram.

c) example [generated audio](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0221/DS/).
More generated audio demos can be found in [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0228_opencpop_ds100_rel.zip).
