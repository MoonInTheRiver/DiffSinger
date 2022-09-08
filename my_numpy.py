# coding=utf8

import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style
import argparse
import json
import os
import sys

import librosa

from inference.opencpop.map import cpop_pinyin2ph_func

from acoustic.tmp_audio import save_wav
from acoustic.tmp_hparams import set_hparams, hparams
from acoustic.tmp_text_encoder import TokenTextEncoder

# import acoustic.tmp_cuda
# import torch


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


provider = None


class TestAllInfer:
    def __init__(self, hparams):
        self.hparams = hparams

        phone_list = ["AP", "SP", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f",
                      "g",
                      "h", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l", "m", "n",
                      "o",
                      "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo",
                      "v",
                      "van", "ve", "vn", "w", "x", "y", "z", "zh"]
        self.ph_encoder = TokenTextEncoder(
            None, vocab_list=phone_list, replace_oov=',')
        self.pinyin2phs = cpop_pinyin2ph_func()
        self.spk_map = {'opencpop': 0}

        print("load pe")
        self.pe2 = ort.InferenceSession(
            f"{onnx_dir}/xiaoma_pe.onnx", providers=[provider])
        print("load hifigan")
        self.vocoder2 = ort.InferenceSession(
            f"{onnx_dir}/hifigan.onnx", providers=[provider])
        print("load singer_fs")
        self.model2 = ort.InferenceSession(
            f"{onnx_dir}/singer_fs.onnx", providers=[provider])
        ips = self.model2.get_inputs()
        print(len(ips))
        for i in range(0, len(ips)):
            print(f'{i}. {ips[i].name}')

        print("load singer_denoise")
        self.model3 = ort.InferenceSession(
            f"{onnx_dir}/singer_denoise.onnx", providers=[provider])
        ips = self.model3.get_inputs()
        print(len(ips))
        for i in range(0, len(ips)):
            print(f'{i}. {ips[i].name}')

        print("load over")

    def run_vocoder(self, c, **kwargs):
        # c = c.transpose(2, 1)  # [B, 80, T]
        c = np.transpose(c, (0, 2, 1))
        f0 = kwargs.get('f0')  # [B, T]

        if f0 is not None and hparams.get('use_nsf'):
            ort_inputs = {
                'x': c,
                'f0': f0
            }
        else:
            ort_inputs = {
                'x': c,
                'f0': {}
            }
            # [T]

        ort_out = self.vocoder2.run(None, ort_inputs)
        y = ort_out[0]

        return y[None]

    def preprocess_word_level_input(self, inp):
        # Pypinyin can't solve polyphonic words
        text_raw = inp['text'].replace('最长', '最常').replace('长睫毛', '常睫毛') \
            .replace('那么长', '那么常').replace('多长', '多常') \
            .replace('很长', '很常')  # We hope someone could provide a better g2p module for us by opening pull requests.

        # lyric
        pinyins = lazy_pinyin(text_raw, strict=False)
        ph_per_word_lst = [self.pinyin2phs[pinyin.strip()]
                           for pinyin in pinyins if pinyin.strip() in self.pinyin2phs]

        # Note
        note_per_word_lst = [x.strip()
                             for x in inp['notes'].split('|') if x.strip() != '']
        mididur_per_word_lst = [
            x.strip() for x in inp['notes_duration'].split('|') if x.strip() != '']

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(len(ph_per_word_lst), len(
                note_per_word_lst), len(mididur_per_word_lst))
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            # for phs in one word:
            # single ph like ['ai']  or multiple phs like ['n', 'i']
            ph_in_this_word = ph_per_word.split()

            # for notes in one word:
            # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            # process for the model input
            # Step 1.
            #  Deal with note of 'not slur' case or the first note of 'slur' case
            #  j        ie
            #  F#4/Gb4  F#4/Gb4
            #  0        0
            for ph in ph_in_this_word:
                ph_lst.append(ph)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            # step 2.
            #  Deal with the 2nd, 3rd... notes of 'slur' case
            #  j        ie         ie
            #  F#4/Gb4  F#4/Gb4    C#4/Db4
            #  0        0          1
            # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
            if len(note_in_this_word) > 1:
                for idx in range(1, len(note_in_this_word)):
                    ph_lst.append(ph_in_this_word[-1])
                    note_lst.append(note_in_this_word[idx])
                    midi_dur_lst.append(midi_dur_in_this_word[idx])
                    is_slur.append(1)
        ph_seq = ' '.join(ph_lst)

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print(len(ph_lst), len(note_lst), len(midi_dur_lst))
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = np.array(inp['is_slur_seq'].split(), 'float')
        ph_dur = None
        if inp['ph_dur'] is not None:
            ph_dur = np.array(inp['ph_dur'].split(), 'float')
            print(len(note_lst), len(ph_seq.split()),
                  len(midi_dur_lst), len(ph_dur))
            if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst) == len(ph_dur):
                print('Pass word-notes check.')
            else:
                print('The number of words does\'t match the number of notes\' windows. ',
                      'You should split the note(s) for each word by | mark.')
                return None
        else:
            print('Automatic phone duration mode')
            print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
            if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
                print('Pass word-notes check.')
            else:
                print('The number of words does\'t match the number of notes\' windows. ',
                      'You should split the note(s) for each word by | mark.')
                return None
        return ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur

    def preprocess_input(self, inp, input_type='word'):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'opencpop')

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp)
        # like transcriptions.txt in Opencpop dataset.
        elif input_type == 'phoneme':
            ret = self.preprocess_phoneme_level_input(inp)
        else:
            print('Invalid input type.')
            return None

        if ret:
            if input_type == 'word':
                ph_seq, note_lst, midi_dur_lst, is_slur = ret
            else:
                ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur = ret
        else:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                     for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'ph_dur': None}
        item['ph_len'] = len(item['ph_token'])
        if input_type == 'phoneme':
            item['ph_dur'] = ph_dur
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = np.array(item['ph_token'], np.int64)[None, :]
        # txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = np.array([txt_tokens.shape[1]], np.int64)
        # txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = np.zeros(item['spk_id'], np.int64)[None, :]
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        pitch_midi = np.array(item['pitch_midi'], np.int64)[
            None, :hparams['max_frames']]
        # pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = np.array(item['midi_dur'], np.float32)[
            None, :hparams['max_frames']]
        # midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = np.array(item['is_slur'], np.int64)[
            None, :hparams['max_frames']]
        # is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)
        mel2ph = None

        # if item['ph_dur'] is not None:
        #     ph_acc = np.around(np.add.accumulate(24000 * item['ph_dur'] / 128)).astype('int')
        #     ph_dur = np.diff(ph_acc, prepend=0)
        #     ph_dur = np.array(ph_dur, np.int64)[None, :hparams['max_frames']]
        #     lr = LengthRegulator()
        #     mel2ph = lr(ph_dur, txt_tokens == 0).detach()

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur,
            'mel2ph': mel2ph
        }
        return batch

    def forward_model(self, inp):

        print("[Status] Preprocess")

        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        mel2ph = sample['mel2ph']

        mel2ph = None

        print("[Status] Run fs")

        decoder_inp = self.model2.run(
            None,
            {
                "txt_tokens": txt_tokens,
                # "spk_id": spk_id,
                "pitch_midi": sample['pitch_midi'],
                "midi_dur": sample['midi_dur'],
                "is_slur": sample['is_slur'],
                # "mel2ph": np.array([0, 0]).astype(np.int64)
            }
        )
        cond = np.transpose(decoder_inp[0], (0, 2, 1))
        # cond = torch.from_numpy(decoder_inp[0]).transpose(1, 2)

        t = hparams['K_step']
        # print('===> gaussion start.')
        shape = (cond.shape[0], 1,
                 hparams['audio_num_mel_bins'], cond.shape[2])
        # x = torch.randn(shape)
        # x = torch.zeros(shape, device=device)
        x = np.random.randn(*shape).astype(np.float32)

        print("[Status] Run sample")

        for i in tqdm(reversed(range(0, t)), desc='[Status] Sample step', total=t):
            res2 = self.model3.run(
                None,
                {
                    "x": x,
                    "t": np.array([i]).astype(np.int64),
                    "cond": cond,
                }
            )
            x = res2[0]

        # x = x[:, 0].transpose(1, 2)
        x = np.transpose(x[:, 0], (0, 2, 1))

        if mel2ph is not None:  # for singing
            mel_out = denorm_spec(x) * ((mel2ph > 0).float()[:, :, None])
        else:
            mel_out = denorm_spec(x)

        # mel_out = output['mel_out']  # [B, T,80]

        print("[Status] Run pe")

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            pe2_res = self.pe2.run(None,
                                   {
                                       'mel_input': mel_out
                                   }
                                   )

            # pe predict from Pred mel
            f0_pred = pe2_res[1]
        else:
            # f0_pred = output['f0_denorm']
            f0_pred = None

        print("[Status] Run vocoder")

        # Run Vocoder
        wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        # wav_out = wav_out.cpu().numpy()
        return wav_out[0]

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(
            inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output


root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

onnx_dir = f'{root_dir}/acoustic/models'

parser = argparse.ArgumentParser(description='Run DiffSinger inference')
parser.add_argument('proj', type=str, help='Path to the input file')

parser.add_argument('-o', '--out', type=str, default='./infer_out',
                    required=False, help='Path of the output folder')

parser.add_argument('-t', '--title', type=str, required=False,
                    help='Title of output file')

parser.add_argument('-d', '--device', type=str,
                    help='Use gpu to synthesize', default='cpu')

args = parser.parse_args()

use_gpu = args.device == 'gpu'

provider = ('CUDAExecutionProvider', {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}) if use_gpu else "CPUExecutionProvider"


sys.argv = [
    f'{root_dir}/inference/ds_e2e.py',
    '--config',
    f'{root_dir}/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0228_opencpop_ds100_rel'
]

spec_max = None
spec_min = None


if __name__ == '__main__':
    with open(args.proj, 'r', encoding='utf-8') as f:
        c = json.load(f)

    name = os.path.basename(args.proj).split(
        '.')[0] if not args.title else args.title
    target = os.path.join(args.out, f'{name}.wav')

    set_hparams(print_hparams=False)

    spec_min = np.array(hparams['spec_min'], np.float32)[
        None, None, :hparams['keep_bins']]
    spec_max = np.array(hparams['spec_max'], np.float32)[
        None, None, :hparams['keep_bins']]

    infer_ins = TestAllInfer(hparams)

    out = infer_ins.infer_once(c)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f'[Status] Save audio: {target}')
    save_wav(out, target, hparams['audio_sample_rate'])

    print("OK")
