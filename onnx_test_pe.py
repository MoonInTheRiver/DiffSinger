# coding=utf8

import os
import sys
import inference.svs.ds_e2e as e2e
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

import numpy as np

import torch
import onnxruntime as ort

root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

sys.argv = [
    f'{root_dir}/inference/svs/ds_e2e.py',
    '--config',
    f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0228_opencpop_ds100_rel'
]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class TestHifiganInfer(e2e.DiffSingerE2EInfer):
    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)

        self.pe2 = ort.InferenceSession("xiaoma_pe.onnx")
        self.vocoder2 = ort.InferenceSession("hifigan.onnx")

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)  # [B, 80, T]
        f0 = kwargs.get('f0')  # [B, T]

        if f0 is not None and hparams.get('use_nsf'):
            ort_inputs = {
                'x': to_numpy(c),
                'f0': to_numpy(f0)
            }
        else:
            ort_inputs = {
                'x': to_numpy(c),
                'f0': {}
            }
            # [T]

        ort_out = self.vocoder2.run(None, ort_inputs)
        y = torch.from_numpy(ort_out[0]).to(self.device)

        return y[None]

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')

        print(txt_tokens.shape)
        print(spk_id.shape)
        print(sample['pitch_midi'].shape)
        print(sample['midi_dur'].shape)
        if (sample['is_slur'] is not None):
            print(sample['is_slur'].shape)
        if (sample['mel2ph'] is not None):
            print(sample['mel2ph'].shape)

        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'], mel2ph=sample['mel2ph'])

            mel_out = output['mel_out']  # [B, T,80]

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                pe2_res = self.pe2.run(None, 
                    {
                        'mel_input': to_numpy(mel_out)
                    }
                )
                
                # pe predict from Pred mel
                f0_pred = torch.from_numpy(pe2_res[1])

            else:
                f0_pred = output['f0_denorm']

            # Run Vocoder
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    c = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
    }  # user input: Chinese characters

    target = "./infer_out/onnx_test_res.wav"

    set_hparams(print_hparams=False)
    infer_ins = TestHifiganInfer(hparams)

    out = infer_ins.infer_once(c)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f'| save audio: {target}')
    save_wav(out, target, hparams['audio_sample_rate'])

    print(infer_ins.pe)
    print("OK")
