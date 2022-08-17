# coding=utf8

import os
import sys
import inference.svs.ds_e2e as e2e
from modules.fastspeech.pe import PitchExtractor
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from utils import load_ckpt
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

import acoustic.dfs_models as adm

import torch
import numpy as np

from utils.text_encoder import TokenTextEncoder
from usr.diffsinger_task import DIFF_DECODERS

root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

sys.argv = [
    f'{root_dir}/inference/svs/ds_e2e.py',
    '--config',
    f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0228_opencpop_ds100_rel'
]


class GaussianDiffusionWrap(adm.GaussianDiffusionFS):
    def forward(self, txt_tokens,
                # Wrapped Arguments
                spk_id,
                pitch_midi,
                midi_dur,
                is_slur,
                mel2ph,
                ):

        print(f"txt_tokens: {txt_tokens}")
        print(f"spk_id: {spk_id}")
        print(f"pitch_midi: {pitch_midi}")
        print(f"midi_dur: {midi_dur}")
        print(f"is_slur: {is_slur}")
        print(f"mel2ph: {mel2ph}")

        if (mel2ph[0].item() == 0):
            mel2ph = None
        else:
            mel2ph = mel2ph[1].item()
        
        if (torch.numel(txt_tokens) == 0):
            txt_tokens = None
        if (torch.numel(spk_id) == 0):
            spk_id = None
        if (torch.numel(pitch_midi) == 0):
            pitch_midi = None
        if (torch.numel(midi_dur) == 0):
            midi_dur = None
        if (torch.numel(is_slur) == 0):
            is_slur = None

        return super().forward(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                               pitch_midi=pitch_midi, midi_dur=midi_dur,
                               is_slur=is_slur, mel2ph=mel2ph)


class DFSInferWrapped(e2e.DiffSingerE2EInfer):
    def build_model(self):
        model = GaussianDiffusionWrap(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](
                hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()

        return model


class DFSInferWrapped2(e2e.DiffSingerE2EInfer):
    def build_model(self):
        model = adm.GaussianDiffusionDenoise(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](
                hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()

        return model


if __name__ == '__main__':

    inp = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
    }  # user input: Chinese characters

    set_hparams(print_hparams=False)

    dev = 'cuda'

    infer_ins = DFSInferWrapped(hparams)
    infer_ins.model.to(dev)

    infer_ins2 = DFSInferWrapped2(hparams)
    infer_ins2.model.to(dev)

    adm.device = dev

    with torch.no_grad():
        inp = infer_ins.preprocess_input(
            inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        sample = infer_ins.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')

        print(txt_tokens)
        print(spk_id)
        print(sample['pitch_midi'])
        print(sample['midi_dur'])
        print(sample['is_slur'])
        print(sample['mel2ph'])

        torch.onnx.export(
            infer_ins.model,
            (
                txt_tokens.to(dev),
                # {
                #     'spk_id': spk_id.to(dev),
                #     'pitch_midi': sample['pitch_midi'].to(dev),
                #     'midi_dur': sample['midi_dur'].to(dev),
                #     'is_slur': spk_id.to(dev),
                #     'mel2ph': spk_id.to(dev)
                # }
                spk_id.to(dev),
                sample['pitch_midi'].to(dev),
                sample['midi_dur'].to(dev),
                sample['is_slur'].to(dev),
                torch.from_numpy(np.array([0, 0]).astype(np.int64)).to(dev),
            ),
            "singer_fs.onnx",
            # verbose=True,
            input_names=["txt_tokens", "spk_id",
                         "pitch_midi", "midi_dur", "is_slur", "mel2ph"],
            dynamic_axes={
                "txt_tokens": {
                    0: "a",
                    1: "b",
                },
                "spk_id": {
                    0: "a",
                    1: "b",
                },
                "pitch_midi": {
                    0: "a",
                    1: "b",
                },
                "midi_dur": {
                    0: "a",
                    1: "b",
                },
                "is_slur": {
                    0: "a",
                    1: "b",
                }
            },
            opset_version=11
        )

        # fs_res = infer_ins.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
        #                          pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
        #                          is_slur=sample['is_slur'], mel2ph=sample['mel2ph'])
        # cond = fs_res.transpose(1, 2)
        # shape = (cond.shape[0], 1, infer_ins.model.mel_bins, cond.shape[2])
        # x = torch.randn(shape, device=dev)

        torch.onnx.export(
            infer_ins2.model,
            (
                torch.rand(1, 1, 80, 967).to(dev),
                torch.full((1,), 1, dtype=torch.long).to(dev),
                torch.rand(1, 256, 967).to(dev),
            ),
            "singer_denoise.onnx",
            input_names=[
                "x",
                "t",
                "cond",
            ],
            dynamic_axes={
                "x": {
                    0: "batch_size",
                    2: "num_mel_bin",
                    3: "frames",
                },
                "cond": {
                    0: "batch_size",
                    1: "what",
                    2: "frames",
                }
            },
            opset_version=11
        )

    print("OK")
