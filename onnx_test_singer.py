# coding=utf8

import os
from pyexpat import model
import sys
import inference.svs.ds_e2e as e2e
from inference.svs.opencpop.map import cpop_pinyin2ph_func
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

import numpy as np

import torch
import onnxruntime as ort

from tqdm import tqdm

from utils.text_encoder import TokenTextEncoder

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
    if (tensor is None):
        return np.array([[]])
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


spec_max = 0
spec_min = 0


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


class TestAllInfer(e2e.DiffSingerE2EInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device

        phone_list = ["AP", "SP", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g",
                      "h", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l", "m", "n", "o",
                      "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v",
                      "van", "ve", "vn", "w", "x", "y", "z", "zh"]
        self.ph_encoder = TokenTextEncoder(
            None, vocab_list=phone_list, replace_oov=',')
        self.pinyin2phs = cpop_pinyin2ph_func()
        self.spk_map = {'opencpop': 0}

        print("load pe")
        self.pe2 = ort.InferenceSession("xiaoma_pe.onnx")
        print("load hifigan")
        self.vocoder2 = ort.InferenceSession("hifigan.onnx")
        print("load singer_fs")
        self.model2 = ort.InferenceSession("singer_fs.onnx")
        ips = self.model2.get_inputs()
        print(len(ips))
        for i in range(0, len(ips)):
            print(f'{i}. {ips[i].name}')

        print("load singer_denoise")
        self.model3 = ort.InferenceSession("singer_denoise.onnx")
        ips = self.model3.get_inputs()
        print(len(ips))
        for i in range(0, len(ips)):
            print(f'{i}. {ips[i].name}')

        print("load over")

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
        mel2ph = sample['mel2ph']

        device = txt_tokens.device

        with torch.no_grad():
            decoder_inp = self.model2.run(
                None,
                {
                    "txt_tokens": to_numpy(txt_tokens),
                    # "spk_id": to_numpy(spk_id),
                    "pitch_midi": to_numpy(sample['pitch_midi']).astype(np.int64),
                    "midi_dur": to_numpy(sample['midi_dur']),
                    "is_slur": to_numpy(sample['is_slur']).astype(np.int64),
                    # "mel2ph": np.array([0, 0]).astype(np.int64)
                }
            )

            cond = torch.from_numpy(decoder_inp[0]).transpose(1, 2)

            print(f'cond2: {cond}')

            t = hparams['K_step']
            print('===> gaussion start.')
            shape = (cond.shape[0], 1,
                     hparams['audio_num_mel_bins'], cond.shape[2])
            x = torch.randn(shape, device=device)
            # x = torch.zeros(shape, device=device)

            for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                res2 = self.model3.run(
                    None,
                    {
                        "x": to_numpy(x),
                        "t": np.array([i]).astype(np.int64),
                        "cond": to_numpy(cond),
                    }
                )
                x = torch.from_numpy(res2[0])
                cond = torch.from_numpy(res2[1])

            x = x[:, 0].transpose(1, 2)

            if mel2ph is not None:  # for singing
                mel_out = denorm_spec(x) * ((mel2ph > 0).float()[:, :, None])
            else:
                mel_out = denorm_spec(x)

            # mel_out = output['mel_out']  # [B, T,80]

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                pe2_res = self.pe2.run(None,
                                       {
                                           'mel_input': to_numpy(mel_out)
                                       }
                                       )

                # pe predict from Pred mel
                f0_pred = torch.from_numpy(pe2_res[1])

            else:
                # f0_pred = output['f0_denorm']
                f0_pred = None

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

    target = "./infer_out/onnx_test_singer_res.wav"

    set_hparams(print_hparams=False)

    spec_min= torch.FloatTensor(hparams['spec_min'])[None, None, :hparams['keep_bins']]
    spec_max= torch.FloatTensor(hparams['spec_max'])[None, None, :hparams['keep_bins']]

    infer_ins = TestAllInfer(hparams)

    out = infer_ins.infer_once(c)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f'| save audio: {target}')
    save_wav(out, target, hparams['audio_sample_rate'])

    print("OK")
