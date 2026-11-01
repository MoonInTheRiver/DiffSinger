import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
    AdamWLinear,
)
from modules.duration import DurationRatioPredictor
from modules.duration.word_groups import word_budget
from modules.fastspeech.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX

DURATION_PREDICTOR_ARCHS = {
    'fs2': DurationPredictor,
    'resnet': DurationPredictor,
    'attn': DurationRatioPredictor,
}


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.linguistic_mode = 'word' if hparams['predict_dur'] else 'phoneme'
        self.use_lang_id = hparams['use_lang_id']
        self.use_variance_scaling = hparams.get('use_variance_scaling', False)
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        if self.use_lang_id:
            self.lang_embed = Embedding(hparams['num_lang'] + 1, hparams['hidden_size'], padding_idx=0)

        if self.predict_dur:
            self.onset_embed = Embedding(2, hparams['hidden_size'])
            self.word_dur_embed = AdamWLinear(1, hparams['hidden_size'])
        else:
            self.ph_dur_embed = AdamWLinear(1, hparams['hidden_size'])

        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams.get('rel_pos', False), 
            use_rope=hparams.get('use_rope', False), rope_interleaved=hparams.get('rope_interleaved', True),
            rope_theta=hparams.get('rope_theta', 10000)
        )

        dur_hparams = hparams['dur_prediction_args']
        if self.predict_dur:
            self.midi_embed = Embedding(128, hparams['hidden_size'])
            self.dur_arch = dur_hparams['arch']
            if self.dur_arch not in DURATION_PREDICTOR_ARCHS:
                raise NotImplementedError(f"unsupported duration predictor architecture: {self.dur_arch}")
            self.dur_predictor = DURATION_PREDICTOR_ARCHS[self.dur_arch].from_hparams(
                in_dims=hparams['hidden_size'],
                dur_hparams=dur_hparams
            )

    def dur_word_budget(self, ph_dur, ph2word, word_dur, infer):
        """Use score durations for inference and summed phoneme durations for training."""
        if self.dur_arch != 'attn':
            return None
        if infer:
            return word_dur
        return word_budget(ph_dur, ph2word)

    def forward(
            self, txt_tokens, midi, ph2word,
            ph_dur=None, word_dur=None,
            spk_embed=None, languages=None,
            infer=True
    ):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train, [infer]) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param spk_embed: (train) [B, T_ph, H]
        :param languages (train, infer) [B, T_ph]
        :param infer: whether inference
        :return: encoder_out, ph_dur_pred
        """
        txt_embed = self.txt_embed(txt_tokens)
        if self.linguistic_mode == 'word':
            b = txt_tokens.shape[0]
            onset = torch.diff(ph2word, dim=1, prepend=ph2word.new_zeros(b, 1)) > 0
            onset_embed = self.onset_embed(onset.long())  # [B, T_ph, H]

            if word_dur is None or not infer:
                word_dur = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
                    1, ph2word, ph_dur
                )[:, 1:]  # [B, T_ph] => [B, T_w]
            ph_word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 1, ph2word)  # [B, T_w] => [B, T_ph]
            word_dur_embed = self.word_dur_embed(ph_word_dur.float()[:, :, None])
            extra_embed = onset_embed + word_dur_embed
        elif self.use_variance_scaling:
            extra_embed = self.ph_dur_embed(torch.log(1 + ph_dur.float())[:, :, None])
        else:
            extra_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
            
        if self.use_lang_id:
            lang_embed = self.lang_embed(languages)
            extra_embed += lang_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0)

        if self.predict_dur:
            midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
            dur_cond = encoder_out + midi_embed
            if spk_embed is not None:
                dur_cond += spk_embed
            if self.dur_arch == 'attn':
                ph_dur_pred = self.dur_predictor(
                    dur_cond, x_masks=txt_tokens == PAD_INDEX,
                    ph2word=ph2word,
                    word_budget=self.dur_word_budget(ph_dur, ph2word, word_dur, infer)
                )
            else:
                ph_dur_pred = self.dur_predictor(dur_cond, x_masks=txt_tokens == PAD_INDEX, infer=infer)

            return encoder_out, ph_dur_pred
        else:
            return encoder_out, None


class MelodyEncoder(nn.Module):
    def __init__(self, enc_hparams: dict):
        super().__init__()

        def get_hparam(key):
            return enc_hparams.get(key, hparams.get(key))

        # MIDI inputs
        hidden_size = get_hparam('hidden_size')
        self.use_variance_scaling = hparams.get('use_variance_scaling', False)
        self.note_midi_embed = AdamWLinear(1, hidden_size)
        self.note_dur_embed = AdamWLinear(1, hidden_size)

        # ornament inputs
        self.use_glide_embed = hparams['use_glide_embed']
        self.glide_embed_scale = hparams['glide_embed_scale']
        if self.use_glide_embed:
            # 0: none, 1: up, 2: down
            self.note_glide_embed = Embedding(len(hparams['glide_types']) + 1, hidden_size, padding_idx=0)

        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size, num_layers=get_hparam('enc_layers'),
            ffn_kernel_size=get_hparam('enc_ffn_kernel_size'), ffn_act=get_hparam('ffn_act'),
            dropout=get_hparam('dropout'), num_heads=get_hparam('num_heads'),
            use_pos_embed=get_hparam('use_pos_embed'), rel_pos=get_hparam('rel_pos'),
            use_rope=get_hparam('use_rope'), rope_interleaved=hparams.get('rope_interleaved', True),
            rope_theta=hparams.get('rope_theta', 10000)
        )
        self.out_proj = Linear(hidden_size, hparams['hidden_size'])

    def forward(self, note_midi, note_rest, note_dur, glide=None):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param note_dur: int64 [B, T_n]
        :param glide: int64 [B, T_n]
        :return: [B, T_n, H]
        """
        if self.use_variance_scaling:
            midi_embed = self.note_midi_embed(note_midi[:, :, None] / 128)
            dur_embed = self.note_dur_embed(torch.log(1 + note_dur.float())[:, :, None])
        else:
            midi_embed = self.note_midi_embed(note_midi[:, :, None])
            dur_embed = self.note_dur_embed(note_dur.float()[:, :, None])
        midi_embed *= ~note_rest[:, :, None]
        ornament_embed = 0
        if self.use_glide_embed:
            ornament_embed += self.note_glide_embed(glide) * self.glide_embed_scale
        encoder_out = self.encoder(
            midi_embed, dur_embed + ornament_embed,
            padding_mask=note_midi < 0
        )
        encoder_out = self.out_proj(encoder_out)
        return encoder_out
