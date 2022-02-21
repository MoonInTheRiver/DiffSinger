# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)

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

```
.
|--data
    |--raw
        |--opencpop
            |--segments
                |--transcriptions.txt
                |--wavs
|--checkpoints
    |--0219_opencpop_fs_midi (optional)
    |--0220_opencpop_ds58_midi (optional)
    |--0109_hifigan_bigpopcs_hop128
        |--model_ckpt_steps_1512000.ckpt
        |--config.yaml
```

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the [pre-trained model](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0219_opencpop_fs_midi.zip), or train FFT-Singer from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/opencpop_aux.yaml --exp_name 0219_opencpop_fs_midi --reset
```

Then, to train DiffSinger, run:

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/opencpop_ds58.yaml --exp_name 0220_opencpop_ds58_midi --reset  
```

Remember to adjust the "fs2_ckpt" parameter in `usr/configs/midi/cascade/opencs/opencpop_ds58.yaml` to fit your path.

### 3. Inference Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/cascade/opencs/opencpop_ds58.yaml --exp_name 0128_opencpop_ds58_midi --reset --infer
```

We also provide:
 - the pre-trained model of [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0220_opencpop_ds58_midi.zip);
 - the pre-trained model of [FFT-Singer](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0219_opencpop_fs_midi.zip);

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Some issues.
a) the [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) is trained on our [vocoder dataset](https://dl.acm.org/doi/abs/10.1145/3474085.3475437) and the training set of [PopCS](https://arxiv.org/abs/2105.02446). Opencpop is the out-of-domain dataset (unseen speaker). This may cause the deterioration of audio quality, and we are considering fine-tuning this vocoder on the training set of Opencpop.

b) in this version of codes, we used the melody frontend ([lyric + MIDI]->[F0+ph_dur]) to predict F0 contour and phoneme duration.

c) example [generated audio](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0221/0220_opencpop_ds58_midi/).
More generated audio demos can be found in [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0220_opencpop_ds58_midi.zip) or [FFT-Singer](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0219_opencpop_fs_midi.zip).
