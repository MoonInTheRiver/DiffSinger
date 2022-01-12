import os
import random
from copy import deepcopy
import pandas as pd
import logging
from tqdm import tqdm
import json
import glob
from resemblyzer import VoiceEncoder
import traceback
import numpy as np
import pretty_midi
import librosa
from scipy.interpolate import interp1d

from utils.hparams import hparams
from data_gen.tts.data_gen_utils import build_phone_encoder
from utils.pitch_utils import f0_to_coarse
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.binarizer_zh import ZhBinarizer
from vocoders.base_vocoder import VOCODERS


def split_train_test_set(item_names):
    item_names = deepcopy(item_names)
    test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    logging.info("train {}".format(len(train_item_names)))
    logging.info("test {}".format(len(test_item_names)))
    return train_item_names, test_item_names


class SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2f0fn = {}
        self.item2tgfn = {}
        self.item2spk = {}

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            wav_suffix = '_wf0.wav'
            txt_suffix = '.txt'
            ph_suffix = '_ph.txt'
            tg_suffix = '.TextGrid'
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*{wav_suffix}')

            for piece_path in all_wav_pieces:
                item_name = raw_item_name = piece_path[len(processed_data_dir)+1:].replace('/', '-')[:-len(wav_suffix)]
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = open(f'{piece_path.replace(wav_suffix, txt_suffix)}').readline()
                self.item2ph[item_name] = open(f'{piece_path.replace(wav_suffix, ph_suffix)}').readline()
                self.item2wavfn[item_name] = piece_path

                self.item2spk[item_name] = 'SPK1'
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
                self.item2tgfn[item_name] = piece_path.replace(wav_suffix, tg_suffix)

        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for ph_sent in self.item2ph.values():
                ph_set += ph_sent.split(' ')
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

    # @staticmethod
    # def get_pitch(wav_fn, spec, res):
    #     wav_suffix = '_wf0.wav'
    #     f0_suffix = '_f0.npy'
    #     f0fn = wav_fn.replace(wav_suffix, f0_suffix)
    #     pitch_info = np.load(f0fn)
    #     f0 = [x[1] for x in pitch_info]
    #     spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
    #     f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
    #     f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]
    #     # f0_x_coor = np.arange(0, 1, 1 / len(f0))
    #     # f0_x_coor[-1] = 1
    #     # f0 = interp1d(f0_x_coor, f0, 'nearest')(spec_x_coor)[:len(spec)]
    #     if sum(f0) == 0:
    #         raise BinarizationError("Empty f0")
    #     assert len(f0) == len(spec), (len(f0), len(spec))
    #     pitch_coarse = f0_to_coarse(f0)
    #
    #     # vis f0
    #     # import matplotlib.pyplot as plt
    #     # from textgrid import TextGrid
    #     # tg_fn = wav_fn.replace(wav_suffix, '.TextGrid')
    #     # fig = plt.figure(figsize=(12, 6))
    #     # plt.pcolor(spec.T, vmin=-5, vmax=0)
    #     # ax = plt.gca()
    #     # ax2 = ax.twinx()
    #     # ax2.plot(f0, color='red')
    #     # ax2.set_ylim(0, 800)
    #     # itvs = TextGrid.fromFile(tg_fn)[0]
    #     # for itv in itvs:
    #     #     x = itv.maxTime * hparams['audio_sample_rate'] / hparams['hop_size']
    #     #     plt.vlines(x=x, ymin=0, ymax=80, color='black')
    #     #     plt.text(x=x, y=20, s=itv.mark, color='black')
    #     # plt.savefig('tmp/20211229_singing_plots_test.png')
    #
    #     res['f0'] = f0
    #     res['pitch'] = pitch_coarse

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                # cls.get_pitch(wav_fn, mel, res)
                cls.get_pitch(wav, mel, res)
            if binarization_args['with_txt']:
                try:
                    # print(ph)
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class MidiSingingBinarizer(SingingBinarizer):
    @staticmethod
    def get_pitch(wav_fn, spec, res):
        wav_suffix = '_wf0.wav'
        midi_suffix = '.mid'

        ## aux f0
        # f0_suffix = '_f0.npy'
        # f0fn = wav_fn.replace(wav_suffix, f0_suffix)
        # pitch_info = np.load(f0fn)
        # f0 = [x[1] for x in pitch_info]
        # spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
        #
        # f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
        # f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]

        ## read midi
        midi_fn = wav_fn.replace(wav_suffix, midi_suffix)
        pm = pretty_midi.PrettyMIDI(midi_fn)
        notes = np.zeros([len(spec)])
        for n in pm.instruments[0].notes:
            sps = hparams['audio_sample_rate'] / hparams['hop_size']
            notes[int(n.start * sps):int(n.end * sps)] = librosa.midi_to_hz(n.pitch)

        # spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
        # note_x_coor = np.arange(0, 1, 1 / len(notes))[:len(notes)]
        # notes = interp1d(note_x_coor, notes, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]

        f0 = notes

        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(f0) == len(spec), (len(f0), len(spec))
        pitch_coarse = f0_to_coarse(f0)

        # # vis f0
        # import matplotlib.pyplot as plt
        # from textgrid import TextGrid
        # tg_fn = wav_fn.replace(wav_suffix, '.TextGrid')
        # fig = plt.figure(figsize=(12, 6))
        # plt.pcolor(spec.T, vmin=-5, vmax=0)
        # ax = plt.gca()
        # ax2 = ax.twinx()
        # ax2.plot(f0, color='red')
        # ax2.plot(notes, color='white')
        # ax2.set_ylim(0, 800)
        # itvs = TextGrid.fromFile(tg_fn)[0]
        # for itv in itvs:
        #     x = itv.maxTime * hparams['audio_sample_rate'] / hparams['hop_size']
        #     plt.vlines(x=x, ymin=0, ymax=80, color='black')
        #     plt.text(x=x, y=20, s=itv.mark, color='black')
        #     plt.savefig('tmp/1231_singing_plots_test.png')

        res['f0'] = f0
        res['pitch'] = pitch_coarse

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, mel, res)
            if binarization_args['with_txt']:
                try:
                    # print(ph)
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class ZhSingingBinarizer(ZhBinarizer, SingingBinarizer):
    pass


if __name__ == "__main__":
    SingingBinarizer().process()
