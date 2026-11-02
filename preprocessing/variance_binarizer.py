import csv
import json
import os
import pathlib

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate

from basics.base_binarizer import BaseBinarizer, BinarizationError
from modules.fastspeech.tts_modules import LengthRegulator
from modules.pe import get_pitch_extractor
from utils.binarizer_utils import (
    dur_sec_to_frames,
    get_mel2ph_torch,
    get_energy_librosa,
    get_breathiness,
    get_voicing,
    get_tension_base_harmonic,
    get_smoothing_conv,
)
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.pitch_utils import interp_f0
from utils.plot import distribution_to_figure

os.environ["OMP_NUM_THREADS"] = "1"
VARIANCE_ITEM_ATTRIBUTES = [
    'spk_id',  # index number of dataset/speaker, int64
    'languages',  # index numbers of phoneme languages, int64[T_ph,]
    'tokens',  # index numbers of phonemes, int64[T_ph,]
    'ph_dur',  # durations of phonemes, in number of frames, int64[T_ph,]
    'midi',  # phoneme-level mean MIDI pitch, int64[T_ph,]
    'ph2word',  # similar to mel2ph format, representing number of phones within each note, int64[T_ph,]
    'mel2ph',  # mel2ph format representing number of frames within each phone, int64[T_s,]
    'note_midi',  # note-level MIDI pitch, float32[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'note_glide',  # flags for glides, 0 = none, 1 = up, 2 = down, int64[T_n,]
    'mel2note',  # mel2ph format representing number of frames within each note, int64[T_s,]
    'base_pitch',  # interpolated and smoothed frame-level MIDI pitch, float32[T_s,]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'uv',  # unvoiced masks (only for objective evaluation metrics), bool[T_s,]
    'energy',  # frame-level RMS (dB), float32[T_s,]
    'breathiness',  # frame-level RMS of aperiodic parts (dB), float32[T_s,]
    'voicing',  # frame-level RMS of harmonic parts (dB), float32[T_s,]
    'tension',  # frame-level tension (logit), float32[T_s,]
]
WAV_CANDIDATE_EXTENSIONS = ['.wav', '.flac']
DS_INDEX_SEP = '#'


class VarianceBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=VARIANCE_ITEM_ATTRIBUTES)

        self.use_glide_embed = hparams['use_glide_embed']
        glide_types = hparams['glide_types']
        assert 'none' not in glide_types, 'Type name \'none\' is reserved and should not appear in glide_types.'
        self.glide_map = {
            'none': 0,
            **{
                typename: idx + 1
                for idx, typename in enumerate(glide_types)
            }
        }

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        predict_voicing = hparams['predict_voicing']
        predict_tension = hparams['predict_tension']
        self.predict_variances = predict_energy or predict_breathiness or predict_voicing or predict_tension
        self.lr = LengthRegulator().to(self.device)
        self.prefer_ds = self.binarization_args['prefer_ds']
        self.cached_ds = {}

    def load_attr_from_ds(self, ds_id, name, attr, idx=0):
        item_name = f'{ds_id}:{name}'
        item_name_with_idx = f'{item_name}{DS_INDEX_SEP}{idx}'
        if item_name_with_idx in self.cached_ds:
            ds = self.cached_ds[item_name_with_idx][0]
        elif item_name in self.cached_ds:
            ds = self.cached_ds[item_name][idx]
        else:
            ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}{DS_INDEX_SEP}{idx}.ds'
            if ds_path.exists():
                cache_key = item_name_with_idx
            else:
                ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}.ds'
                cache_key = item_name
            if not ds_path.exists():
                return None
            with open(ds_path, 'r', encoding='utf8') as f:
                ds = json.load(f)
            if not isinstance(ds, list):
                ds = [ds]
            self.cached_ds[cache_key] = ds
            ds = ds[0] if cache_key == item_name_with_idx else ds[idx]
        return ds.get(attr)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id, spk, lang):
        meta_data_dict = {}

        with open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8') as f:
            for utterance_label in csv.DictReader(f):
                utterance_label: dict
                item_name = utterance_label['name']
                item_base_name, *item_seg_idx = item_name.rsplit(DS_INDEX_SEP, maxsplit=1)
                item_idx = int(item_seg_idx[0]) if item_seg_idx else 0

                def require(attr, optional=False):
                    if self.prefer_ds:
                        value = self.load_attr_from_ds(ds_id, item_base_name, attr, item_idx)
                    else:
                        value = None
                    if value is None:
                        value = utterance_label.get(attr)
                    if value is None and not optional:
                        raise ValueError(f'Missing required attribute {attr} of item \'{item_name}\'.')
                    return value

                wav_fn = None
                for ext in WAV_CANDIDATE_EXTENSIONS:
                    candidate_fn = raw_data_dir / 'wavs' / f'{item_name}{ext}'
                    if candidate_fn.exists():
                        wav_fn = candidate_fn
                        break
                if wav_fn is None and not self.prefer_ds:
                    raise FileNotFoundError(
                        f'Waveform file not found for item \'{item_name}\'. '
                        f'Candidate extensions: {WAV_CANDIDATE_EXTENSIONS}\n'
                        f'If you are using DS files instead of waveform files, please set \'prefer_ds\' to true.'
                    )

                temp_dict = {
                    'ds_idx': item_idx,
                    'spk_id': self.spk_map[spk],
                    'spk_name': spk,
                    'language_id': self.lang_map[lang],
                    'language_name': lang,
                    'wav_fn': str(wav_fn) if wav_fn is not None else None,
                    'lang_seq': [
                        (
                            self.lang_map[lang if '/' not in p else p.split('/', maxsplit=1)[0]]
                            if self.phoneme_dictionary.is_cross_lingual(p if '/' in p else f'{lang}/{p}')
                            else 0
                        )
                        for p in require('ph_seq').split()
                    ],
                    'ph_seq': self.phoneme_dictionary.encode(require('ph_seq'), lang=lang),
                    'ph_dur': [float(x) for x in require('ph_dur').split()],
                    'ph_text': require('ph_seq'),
                }

                assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                    f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                assert all(ph_dur >= 0 for ph_dur in temp_dict['ph_dur']), \
                    f'Negative ph_dur found in \'{item_name}\'.'

                if hparams['predict_dur']:
                    temp_dict['ph_num'] = [int(x) for x in require('ph_num').split()]
                    assert len(temp_dict['ph_seq']) == sum(temp_dict['ph_num']), \
                        f'Sum of ph_num does not equal length of ph_seq in \'{item_name}\'.'

                if hparams['predict_pitch']:
                    temp_dict['note_seq'] = require('note_seq').split()
                    temp_dict['note_dur'] = [float(x) for x in require('note_dur').split()]
                    assert all(note_dur >= 0 for note_dur in temp_dict['note_dur']), \
                        f'Negative note_dur found in \'{item_name}\'.'
                    assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                        f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
                    assert any([note != 'rest' for note in temp_dict['note_seq']]), \
                        f'All notes are rest in \'{item_name}\'.'
                    if hparams['use_glide_embed']:
                        note_glide = require('note_glide', optional=True)
                        if note_glide is None:
                            note_glide = ['none' for _ in temp_dict['note_seq']]
                        else:
                            note_glide = note_glide.split()
                            assert len(note_glide) == len(temp_dict['note_seq']), \
                                f'Lengths of note_seq and note_glide mismatch in \'{item_name}\'.'
                            assert all(g in self.glide_map for g in note_glide), \
                                f'Invalid glide type found in \'{item_name}\'.'
                        temp_dict['note_glide'] = note_glide

                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict

        return meta_data_dict

    def check_coverage(self):
        super().check_coverage()
        if not hparams['predict_pitch']:
            return

        # MIDI pitch distribution summary
        midi_map = {}
        for item_name in self.items:
            for midi in self.items[item_name]['note_seq']:
                if midi == 'rest':
                    continue
                midi = librosa.note_to_midi(midi, round_midi=True)
                if midi in midi_map:
                    midi_map[midi] += 1
                else:
                    midi_map[midi] = 1

        print('===== MIDI Pitch Distribution Summary =====')
        for i, key in enumerate(sorted(midi_map.keys())):
            if i == len(midi_map) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{librosa.midi_to_note(key, unicode=False)}\': {midi_map[key]}', end=end)

        # Draw graph.
        midis = sorted(midi_map.keys())
        notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
        plt = distribution_to_figure(
            title='MIDI Pitch Distribution Summary',
            x_label='MIDI Key', y_label='Number of occurrences',
            items=notes, values=[midi_map.get(m, 0) for m in range(midis[0], midis[-1] + 1)]
        )
        filename = self.binary_data_dir / 'midi_distribution.jpg'
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')

        if self.use_glide_embed:
            # Glide type distribution summary
            glide_count = {
                g: 0
                for g in self.glide_map
            }
            for item_name in self.items:
                for glide in self.items[item_name]['note_glide']:
                    if glide == 'none' or glide not in self.glide_map:
                        glide_count['none'] += 1
                    else:
                        glide_count[glide] += 1

            print('===== Glide Type Distribution Summary =====')
            for i, key in enumerate(sorted(glide_count.keys(), key=lambda k: self.glide_map[k])):
                if i == len(glide_count) - 1:
                    end = '\n'
                elif i % 10 == 9:
                    end = ',\n'
                else:
                    end = ', '
                print(f'\'{key}\': {glide_count[key]}', end=end)

            if any(n == 0 for _, n in glide_count.items()):
                raise BinarizationError(
                    f'Missing glide types in dataset: '
                    f'{sorted([g for g, n in glide_count.items() if n == 0], key=lambda k: self.glide_map[k])}'
                )

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        ds_id, name = item_name.split(':', maxsplit=1)
        name = name.rsplit(DS_INDEX_SEP, maxsplit=1)[0]
        ds_id = int(ds_id)
        ds_seg_idx = meta_data['ds_idx']
        seconds = sum(meta_data['ph_dur'])
        T_ph = len(meta_data['ph_seq'])
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'spk_name': meta_data['spk_name'],
            'seconds': seconds,
            'languages': np.array(meta_data['lang_seq'], dtype=np.int64),
            'tokens': np.array(meta_data['ph_seq'], dtype=np.int64),
            'ph_text': meta_data['ph_text'],
        }

        ph_dur_sec = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
        ph_dur = dur_sec_to_frames(ph_dur_sec, self.timestep, device=self.device)
        ph_acc = ph_dur.cumsum(0)  # frame boundaries, also used by the zero-duration fallback
        length = int(ph_acc[-1])
        processed_input['ph_dur'] = ph_dur.cpu().numpy()
        processed_input['length'] = length

        mel2ph = get_mel2ph_torch(
            self.lr, ph_dur_sec, length, self.timestep, device=self.device
        )

        if hparams['predict_pitch'] or self.predict_variances:
            processed_input['mel2ph'] = mel2ph.cpu().numpy()

        # Below: extract actual f0, convert to pitch and calculate delta pitch
        if meta_data['wav_fn'] is not None:
            waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        else:
            waveform = None

        f0 = uv = None
        if self.prefer_ds:
            f0_seq = self.load_attr_from_ds(ds_id, name, 'f0_seq', idx=ds_seg_idx)
            if f0_seq is not None:
                f0_timestep = float(self.load_attr_from_ds(ds_id, name, 'f0_timestep', idx=ds_seg_idx))
                # Interpolate unvoiced parts before resampling.
                f0_points, uv_points = interp_f0(np.array(f0_seq.split(), np.float32))
                f0 = resample_align_curve(
                    f0_points,
                    original_timestep=f0_timestep,
                    target_timestep=self.timestep,
                    align_length=length
                )
                uv = resample_align_curve(
                    uv_points.astype(np.float32),
                    original_timestep=f0_timestep,
                    target_timestep=self.timestep,
                    align_length=length
                ) > 0.5
        if f0 is None:
            pitch_extractor = get_pitch_extractor(hparams['pe'], hparams['pe_ckpt'], device=self.device)
            f0, uv = pitch_extractor.get_pitch(
                waveform, samplerate=hparams['audio_sample_rate'], length=length,
                hop_size=hparams['hop_size'], f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
                interp_uv=True
            )
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        pitch = torch.from_numpy(librosa.hz_to_midi(f0.astype(np.float32)).astype(np.float32)).to(self.device)

        if hparams['predict_dur']:
            ph_num = torch.LongTensor(meta_data['ph_num']).to(self.device)
            ph2word = self.lr(ph_num[None])[0]
            processed_input['ph2word'] = ph2word.cpu().numpy()
            mel2dur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 0, mel2ph)  # frame-level phone duration
            ph_midi = pitch.new_zeros(T_ph + 1).scatter_add(
                0, mel2ph, pitch / mel2dur
            )[1:]
            # Phones collapsed to 0 frames receive no scatter contribution;
            # fall back to the pitch value at their position on the time axis.
            zero_dur = ph_dur <= 0
            if zero_dur.any():
                bound = torch.cat([ph_acc.new_zeros(1), ph_acc[:-1]]).clamp(min=0, max=length - 1)
                ph_midi[zero_dur] = pitch[bound[zero_dur]]
            processed_input['midi'] = ph_midi.round().long().clamp(min=0, max=127).cpu().numpy()

        if hparams['predict_pitch']:
            # Below: get note sequence and interpolate rest notes
            note_midi = np.array(
                [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in meta_data['note_seq']],
                dtype=np.float32
            )
            note_rest = note_midi < 0
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            note_midi[note_rest] = interp_func(np.where(note_rest)[0])
            processed_input['note_midi'] = note_midi
            processed_input['note_rest'] = note_rest
            note_midi = torch.from_numpy(note_midi).to(self.device)

            note_dur_sec = torch.FloatTensor(meta_data['note_dur']).to(self.device)
            note_dur = dur_sec_to_frames(note_dur_sec, self.timestep, device=self.device)
            processed_input['note_dur'] = note_dur.cpu().numpy()

            mel2note = get_mel2ph_torch(
                self.lr, note_dur_sec, mel2ph.shape[0], self.timestep, device=self.device
            )
            processed_input['mel2note'] = mel2note.cpu().numpy()

            # Below: get ornament attributes
            if hparams['use_glide_embed']:
                processed_input['note_glide'] = np.array([
                    self.glide_map.get(x, 0) for x in meta_data['note_glide']
                ], dtype=np.int64)

            # Below:
            # 1. Get the frame-level MIDI pitch, which is a step function curve
            # 2. smoothen the pitch step curve as the base pitch curve
            frame_midi_pitch = torch.gather(F.pad(note_midi, [1, 0], value=0), 0, mel2note)
            midi_smooth = get_smoothing_conv(
                round(hparams['midi_smooth_width'] / self.timestep), self.device
            )
            smoothed_midi_pitch = midi_smooth(frame_midi_pitch[None])[0]
            processed_input['base_pitch'] = smoothed_midi_pitch.cpu().numpy()

        if hparams['predict_pitch'] or self.predict_variances:
            processed_input['pitch'] = pitch.cpu().numpy()
            processed_input['uv'] = uv

        # Below: extract energy
        if hparams['predict_energy']:
            energy = None
            energy_from_wav = False
            if self.prefer_ds:
                energy_seq = self.load_attr_from_ds(ds_id, name, 'energy', idx=ds_seg_idx)
                if energy_seq is not None:
                    energy = resample_align_curve(
                        np.array(energy_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'energy_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if energy is None:
                energy = get_energy_librosa(
                    waveform, length,
                    hop_size=hparams['hop_size'], win_size=hparams['win_size']
                ).astype(np.float32)
                energy_from_wav = True

            if energy_from_wav:
                energy_smooth = get_smoothing_conv(
                    round(hparams['energy_smooth_width'] / self.timestep), self.device
                )
                energy = energy_smooth(torch.from_numpy(energy).to(self.device)[None])[0].cpu().numpy()

            processed_input['energy'] = energy

        # Only decompose audio when a required curve is missing from the DS file.
        dec_waveform = None

        def get_decomposed_waveform():
            nonlocal dec_waveform
            if dec_waveform is None:
                if waveform is None:
                    raise BinarizationError(f"Missing waveform for '{item_name}' to extract variance curves.")
                dec_waveform = DecomposedWaveform(
                    waveform, samplerate=hparams['audio_sample_rate'], f0=f0 * ~uv,
                    hop_size=hparams['hop_size'], fft_size=hparams['fft_size'], win_size=hparams['win_size'],
                    algorithm=hparams['hnsep']
                )
            return dec_waveform

        # Below: extract breathiness
        if hparams['predict_breathiness']:
            breathiness = None
            breathiness_from_wav = False
            if self.prefer_ds:
                breathiness_seq = self.load_attr_from_ds(ds_id, name, 'breathiness', idx=ds_seg_idx)
                if breathiness_seq is not None:
                    breathiness = resample_align_curve(
                        np.array(breathiness_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'breathiness_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if breathiness is None:
                breathiness = get_breathiness(
                    get_decomposed_waveform(), None, None, length=length
                )
                breathiness_from_wav = True

            if breathiness_from_wav:
                breathiness_smooth = get_smoothing_conv(
                    round(hparams['breathiness_smooth_width'] / self.timestep), self.device
                )
                breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0].cpu().numpy()

            processed_input['breathiness'] = breathiness

        # Below: extract voicing
        if hparams['predict_voicing']:
            voicing = None
            voicing_from_wav = False
            if self.prefer_ds:
                voicing_seq = self.load_attr_from_ds(ds_id, name, 'voicing', idx=ds_seg_idx)
                if voicing_seq is not None:
                    voicing = resample_align_curve(
                        np.array(voicing_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'voicing_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if voicing is None:
                voicing = get_voicing(
                    get_decomposed_waveform(), None, None, length=length
                )
                voicing_from_wav = True

            if voicing_from_wav:
                voicing_smooth = get_smoothing_conv(
                    round(hparams['voicing_smooth_width'] / self.timestep), self.device
                )
                voicing = voicing_smooth(torch.from_numpy(voicing).to(self.device)[None])[0].cpu().numpy()

            processed_input['voicing'] = voicing

        # Below: extract tension
        if hparams['predict_tension']:
            tension = None
            tension_from_wav = False
            if self.prefer_ds:
                tension_seq = self.load_attr_from_ds(ds_id, name, 'tension', idx=ds_seg_idx)
                if tension_seq is not None:
                    tension = resample_align_curve(
                        np.array(tension_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'tension_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if tension is None:
                tension = get_tension_base_harmonic(
                    get_decomposed_waveform(), None, None, length=length, domain='logit'
                )
                tension_from_wav = True

            if tension_from_wav:
                tension_smooth = get_smoothing_conv(
                    round(hparams['tension_smooth_width'] / self.timestep), self.device
                )
                tension = tension_smooth(torch.from_numpy(tension).to(self.device)[None])[0].cpu().numpy()

            processed_input['tension'] = tension

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
