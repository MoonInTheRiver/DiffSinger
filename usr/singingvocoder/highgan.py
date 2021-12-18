import sys

sys.path.append('highgan')

from utils.hparams import hparams
from vocoders.pwg import PWG
import os
from tqdm import tqdm
from utils.indexed_datasets import IndexedDataset
import logging
import torch
import yaml
from vocoders.base_vocoder import BaseVocoder
import models
import numpy as np


def normalize(S):
    return np.clip((S + 100) / 100, -2, 2)


def load_model_config(checkpoint_path, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    model_class = getattr(
        models,
        config.get("generator_type", "ParallelWaveGANGenerator_source"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {checkpoint_path}.")
    model.remove_weight_norm()

    return model, config

class HighGAN(BaseVocoder):
    def __init__(self):
        vocoder_ckpt = hparams['vocoder_ckpt']
        checkpoint_path = "highgan/" + vocoder_ckpt
        config_path = os.path.dirname(checkpoint_path) + "/config.yml"
        model, self.config = load_model_config(checkpoint_path, config_path)
        self.model = model.eval().to('cuda')

    def spec2wav(self, mel, **kwargs):
        f0 = kwargs['f0']
        mel = normalize(20 * mel) * 2
        mel = torch.FloatTensor(mel).transpose(0, 1)[None, ...]
        f0 = torch.FloatTensor(f0)[None, ...]
        pad_fn = torch.nn.ReplicationPad1d(
            self.config["generator_params"].get("aux_context_window", 0))
        device = next(self.model.parameters()).device

        mels = pad_fn(mel).to(device)
        f0 = f0.to(device)
        x = (mels, f0)

        with torch.no_grad():
            y = self.model(*x)
            y = y.view(-1).cpu().numpy()
        return y

    @staticmethod
    def wav2spec(wav_fn, **kwargs):
        return PWG.wav2spec(wav_fn, **kwargs)


if __name__ == '__main__':
    import soundfile as sf

    import matplotlib.pyplot as plt
    def f0_to_figure(f0_gt, mel, i, f0_cwt=None, f0_pred=None):
        fig = plt.figure()
        plt.pcolor(mel.T)
        plt.plot(f0_gt / 10, color='r', label='gt')
        if f0_cwt is not None:
            f0_cwt = f0_cwt.cpu().numpy()
            plt.plot(f0_cwt, color='b', label='cwt')
        if f0_pred is not None:
            f0_pred = f0_pred.cpu().numpy()
            plt.plot(f0_pred, color='green', label='pred')
        plt.legend()
        plt.savefig(f'tmp/highgan_test/{i}_th_f0.png')
        return

    hparams['vocoder_ckpt'] = 'checkpoints/h_2_model/checkpoint-530000steps.pkl'
    vocoder = HighGAN()
    indexed_ds = IndexedDataset(f'data/binary/24k_tmp_parselmouth/test')
    outdir = 'tmp/highgan_test'
    os.makedirs(outdir, exist_ok=True)
    for i in tqdm(range(10)):
        item = indexed_ds[i]
        f0 = item['f0']
        mel = item['mel']
        item_name = item['item_name']
        y = vocoder.spec2wav(mel, f0=f0)
        sf.write(f'{outdir}/{item_name}.wav', y, vocoder.config["sampling_rate"], "PCM_16")
        f0_to_figure(f0, mel, i)