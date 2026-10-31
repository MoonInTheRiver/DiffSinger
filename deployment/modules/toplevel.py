import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deployment.modules.diffusion import (
    GaussianDiffusionONNX, PitchDiffusionONNX, MultiVarianceDiffusionONNX
)
from deployment.modules.rectified_flow import (
    RectifiedFlowONNX, PitchRectifiedFlowONNX, MultiVarianceRectifiedFlowONNX
)
from deployment.modules.fastspeech2 import FastSpeech2AcousticONNX, FastSpeech2VarianceONNX
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from utils.binarizer_utils import SinusoidalSmoothingConv1d
from utils.hparams import hparams


class DiffSingerAcousticONNX(DiffSingerAcoustic):
    def __init__(self, vocab_size, out_dims, cross_lingual_token_idx=None):
        super().__init__(vocab_size, out_dims)
        del self.fs2
        del self.diffusion
        self.fs2 = FastSpeech2AcousticONNX(
            vocab_size=vocab_size,
            cross_lingual_token_idx=cross_lingual_token_idx
        )
        if self.diffusion_type == 'ddpm':
            self.diffusion = GaussianDiffusionONNX(
                out_dims=out_dims,
                num_feats=1,
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        elif self.diffusion_type == 'reflow':
            self.diffusion = RectifiedFlowONNX(
                out_dims=out_dims,
                num_feats=1,
                t_start=hparams['T_start'],
                time_scale_factor=hparams['time_scale_factor'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        else:
            raise ValueError(f"Invalid diffusion type: {self.diffusion_type}")
        self.mel_base = hparams.get('mel_base', '10')

    def ensure_mel_base(self, mel):
        if self.mel_base != 'e':
            # log10 mel to log mel
            mel = mel * 2.30259
        return mel

    def forward_fs2_aux(
            self,
            tokens: Tensor,
            durations: Tensor,
            f0: Tensor,
            variances: dict,
            gender: Tensor = None,
            velocity: Tensor = None,
            spk_embed: Tensor = None,
            languages: Tensor = None
    ):
        condition = self.fs2(
            tokens, durations, f0, variances=variances,
            gender=gender, velocity=velocity, spk_embed=spk_embed,
            languages=languages
        )
        if self.use_shallow_diffusion:
            aux_mel_pred = self.aux_decoder(condition, infer=True)
            return condition, aux_mel_pred
        else:
            return condition

    def forward_shallow_diffusion(
            self, condition: Tensor, x_start: Tensor,
            depth, steps: int
    ) -> Tensor:
        mel_pred = self.diffusion(condition, x_start=x_start, depth=depth, steps=steps)
        return self.ensure_mel_base(mel_pred)

    def forward_diffusion(self, condition: Tensor, steps: int):
        mel_pred = self.diffusion(condition, steps=steps)
        return self.ensure_mel_base(mel_pred)

    def forward_shallow_reflow(
            self, condition: Tensor, x_end: Tensor,
            depth, steps: int
    ):
        mel_pred = self.diffusion(condition, x_end=x_end, depth=depth, steps=steps)
        return self.ensure_mel_base(mel_pred)

    def forward_reflow(self, condition: Tensor, steps: int):
        mel_pred = self.diffusion(condition, steps=steps)
        return self.ensure_mel_base(mel_pred)

    def view_as_fs2_aux(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.diffusion
        model.forward = model.forward_fs2_aux
        return model

    def view_as_diffusion(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        if self.use_shallow_diffusion:
            del model.aux_decoder
            model.forward = model.forward_shallow_diffusion
        else:
            model.forward = model.forward_diffusion
        return model

    def view_as_reflow(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        if self.use_shallow_diffusion:
            del model.aux_decoder
            model.forward = model.forward_shallow_reflow
        else:
            model.forward = model.forward_reflow
        return model


class DiffSingerVarianceONNX(DiffSingerVariance):
    def __init__(self, vocab_size, cross_lingual_token_idx=None):
        super().__init__(vocab_size=vocab_size)
        del self.fs2
        self.fs2 = FastSpeech2VarianceONNX(
            vocab_size=vocab_size,
            cross_lingual_token_idx=cross_lingual_token_idx
        )
        self.hidden_size = hparams['hidden_size']
        if self.predict_pitch:
            del self.pitch_predictor
            self.smooth: nn.Conv1d = None
            pitch_hparams = hparams['pitch_prediction_args']
            if self.diffusion_type == 'ddpm':
                self.pitch_predictor = PitchDiffusionONNX(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    timesteps=hparams['timesteps'],
                    k_step=hparams['K_step'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            elif self.diffusion_type == 'reflow':
                self.pitch_predictor = PitchRectifiedFlowONNX(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    time_scale_factor=hparams['time_scale_factor'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            else:
                raise ValueError(f"Invalid diffusion type: {self.diffusion_type}")
        if self.predict_variances:
            del self.variance_predictor
            if self.diffusion_type == 'ddpm':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceDiffusionONNX)
            elif self.diffusion_type == 'reflow':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceRectifiedFlowONNX)
            else:
                raise NotImplementedError(self.diffusion_type)

    def build_smooth_op(self, device):
        smooth_kernel_size = round(hparams['midi_smooth_width'] * hparams['audio_sample_rate'] / hparams['hop_size'])
        self.smooth = SinusoidalSmoothingConv1d(smooth_kernel_size).eval().to(device)

    def forward_linguistic_encoder_word(self, tokens, word_div, word_dur, languages=None):
        encoder_out, x_masks = self.fs2.forward_encoder_word(tokens, word_div, word_dur, languages=languages)
        return encoder_out, x_masks

    def forward_linguistic_encoder_phoneme(self, tokens, ph_dur, languages=None):
        encoder_out, x_masks = self.fs2.forward_encoder_phoneme(tokens, ph_dur, languages=languages)
        return encoder_out, x_masks

    def forward_dur_predictor(
            self, encoder_out, x_masks, ph_midi, word_div=None, word_dur=None, spk_embed=None
    ):
        if hparams['use_spk_id'] and spk_embed is None and hasattr(self, 'frozen_spk_embed'):
            spk_embed = self.frozen_spk_embed
        return self.fs2.forward_dur_predictor(
            encoder_out, x_masks, ph_midi,
            word_div=word_div, word_dur=word_dur, spk_embed=spk_embed
        )

    def forward_mel2x_gather(self, x_src, x_dur, x_dim=None, check_stretch_embed=False):
        mel2x = self.lr(x_dur)
        _mel2x = mel2x
        if x_dim is not None:
            x_src = F.pad(x_src, [0, 0, 1, 0])
            mel2x = mel2x[..., None].repeat([1, 1, x_dim])
        else:
            x_src = F.pad(x_src, [1, 0])
        x_cond = torch.gather(x_src, 1, mel2x)
        if self.use_stretch_embed and check_stretch_embed:
            stretch = torch.round(1000 * self.sr(_mel2x, x_dur))
            table = self.stretch_embed(torch.arange(0, 1001, device=stretch.device))
            stretch_embed = torch.index_select(table, 0, stretch.view(-1).long()).view_as(x_cond)
            x_cond += stretch_embed
            stretch_embed_rnn_out, _ = self.stretch_embed_rnn(x_cond)
            x_cond += stretch_embed_rnn_out
        return x_cond

    def forward_pitch_preprocess(
            self, encoder_out, ph_dur,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None,
            pitch=None, expr=None, retake=None, spk_embed=None
    ):
        condition = self.forward_mel2x_gather(encoder_out, ph_dur, x_dim=self.hidden_size, check_stretch_embed=True)
        if self.use_melody_encoder:
            if self.melody_encoder.use_glide_embed and note_glide is None:
                note_glide = torch.LongTensor([[0]]).to(encoder_out.device)
            melody_encoder_out = self.melody_encoder(
                note_midi, note_rest, note_dur,
                glide=note_glide
            )
            melody_encoder_out = self.forward_mel2x_gather(melody_encoder_out, note_dur, x_dim=self.hidden_size)
            condition += melody_encoder_out
        if expr is None:
            retake_embed = self.pitch_retake_embed(retake.long())
        else:
            retake_true_embed = self.pitch_retake_embed(
                torch.ones(1, 1, dtype=torch.long, device=encoder_out.device)
            )  # [B=1, T=1] => [B=1, T=1, H]
            retake_false_embed = self.pitch_retake_embed(
                torch.zeros(1, 1, dtype=torch.long, device=encoder_out.device)
            )  # [B=1, T=1] => [B=1, T=1, H]
            expr = (expr * retake)[:, :, None]  # [B, T, 1]
            retake_embed = expr * retake_true_embed + (1. - expr) * retake_false_embed
        pitch_cond = condition + retake_embed
        frame_midi_pitch = self.forward_mel2x_gather(note_midi, note_dur, x_dim=None)
        base_pitch = self.smooth(frame_midi_pitch)
        if self.use_melody_encoder:
            delta_pitch = (pitch - base_pitch) * ~retake
            if self.use_variance_scaling:
                pitch_cond += self.delta_pitch_embed(delta_pitch[:, :, None] / 12)
            else:
                pitch_cond += self.delta_pitch_embed(delta_pitch[:, :, None])
        else:
            base_pitch = base_pitch * retake + pitch * ~retake
            if self.use_variance_scaling:
                pitch_cond += self.base_pitch_embed(base_pitch[:, :, None] / 128)
            else:
                pitch_cond += self.base_pitch_embed(base_pitch[:, :, None])
        if hparams['use_spk_id']:
            if spk_embed is not None:
                pitch_cond += spk_embed
            elif hasattr(self, 'frozen_spk_embed'):
                pitch_cond += self.frozen_spk_embed
        return pitch_cond, base_pitch

    def forward_pitch_reflow(
            self, pitch_cond, steps: int = 10
    ):
        x_pred = self.pitch_predictor(pitch_cond, steps=steps)
        return x_pred

    def forward_pitch_postprocess(self, x_pred, base_pitch):
        pitch_pred = self.pitch_predictor.clamp_spec(x_pred) + base_pitch
        return pitch_pred

    def forward_variance_preprocess(
            self, encoder_out, ph_dur, pitch,
            variances: dict = None, retake=None, spk_embed=None
    ):
        condition = self.forward_mel2x_gather(encoder_out, ph_dur, x_dim=self.hidden_size, check_stretch_embed=True)
        if self.use_variance_scaling:
            variance_cond = condition + self.pitch_embed(pitch[:, :, None] / 12)
        else:
            variance_cond = condition + self.pitch_embed(pitch[:, :, None])
        non_retake_masks = [
            v_retake.float()  # [B, T, 1]
            for v_retake in (~retake).split(1, dim=2)
        ]
        variance_embeds = [
            self.variance_embeds[v_name](variances[v_name][:, :, None] * self.variance_retake_scaling[v_name]) * v_masks
            for v_name, v_masks in zip(self.variance_prediction_list, non_retake_masks)
        ]
        variance_cond += torch.stack(variance_embeds, dim=-1).sum(-1)
        if hparams['use_spk_id']:
            if spk_embed is not None:
                variance_cond += spk_embed
            elif hasattr(self, 'frozen_spk_embed'):
                variance_cond += self.frozen_spk_embed
        return variance_cond

    def forward_variance_reflow(self, variance_cond, steps: int = 10):
        xs_pred = self.variance_predictor(variance_cond, steps=steps)
        return xs_pred

    def forward_variance_postprocess(self, xs_pred):
        if self.variance_predictor.num_feats == 1:
            xs_pred = [xs_pred]
        else:
            xs_pred = xs_pred.unbind(dim=1)
        variance_pred = self.variance_predictor.clamp_spec(xs_pred)
        return tuple(variance_pred)

    def view_as_linguistic_encoder(self):
        model = copy.deepcopy(self)
        if self.predict_pitch:
            del model.pitch_predictor
            if self.use_melody_encoder:
                del model.melody_encoder
        if self.predict_variances:
            del model.variance_predictor
        model.fs2 = model.fs2.view_as_encoder()
        if self.predict_dur:
            model.forward = model.forward_linguistic_encoder_word
        else:
            model.forward = model.forward_linguistic_encoder_phoneme
        return model

    def view_as_dur_predictor(self):
        assert self.predict_dur
        model = copy.deepcopy(self)
        if self.predict_pitch:
            del model.pitch_predictor
            if self.use_melody_encoder:
                del model.melody_encoder
        if self.predict_variances:
            del model.variance_predictor
        model.fs2 = model.fs2.view_as_dur_predictor()
        model.forward = model.forward_dur_predictor
        return model

    def view_as_pitch_preprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_pitch_preprocess
        return model

    def view_as_pitch_predictor(self):
        assert self.predict_pitch
        model = copy.deepcopy(self)
        del model.fs2
        del model.lr
        if self.use_melody_encoder:
            del model.melody_encoder
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_pitch_reflow
        return model

    def view_as_pitch_postprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.use_melody_encoder:
            del model.melody_encoder
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_pitch_postprocess
        return model

    def view_as_variance_preprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
            if self.use_melody_encoder:
                del model.melody_encoder
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_variance_preprocess
        return model

    def view_as_variance_predictor(self):
        assert self.predict_variances
        model = copy.deepcopy(self)
        del model.fs2
        del model.lr
        if self.predict_pitch:
            del model.pitch_predictor
            if self.use_melody_encoder:
                del model.melody_encoder
        model.forward = model.forward_variance_reflow
        return model

    def view_as_variance_postprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
            if self.use_melody_encoder:
                del model.melody_encoder
        model.forward = model.forward_variance_postprocess
        return model
