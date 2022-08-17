from usr.diff.shallow_diffusion_tts import GaussianDiffusion

import torch

device = 'cpu'


class GaussianDiffusionFS(GaussianDiffusion):
    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=True, infer=infer, **kwargs)
        return ret['decoder_inp']


class GaussianDiffusionDenoise(GaussianDiffusion):
    def forward(self, x, t, cond):
        x = self.p_sample(x, t, cond)
        return [x, cond]
