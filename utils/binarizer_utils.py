from functools import lru_cache
from typing import Union

import librosa
import numpy as np
import parselmouth
import torch
from torch.nn import functional as F

from modules.nsf_hifigan.nvSTFT import STFT
from utils.decomposed_waveform import DecomposedWaveform
from utils.pitch_utils import interp_f0


@lru_cache(maxsize=8)
def _get_stft(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device: torch.device):
    """Reuse STFT instances for matching configurations within this process."""
    return STFT(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device=device)


def get_mel_torch(
        waveform, samplerate,
        *,
        num_mel_bins=128, hop_size=512, win_size=2048, fft_size=2048,
        fmin=40, fmax=16000,
        keyshift=0, speed=1, device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    stft = _get_stft(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0), keyshift=keyshift, speed=speed).squeeze(0).T
        return mel_torch.cpu().numpy()


def dur_sec_to_frames(dur_sec, timestep, device='cpu'):
    """Convert durations in seconds into integer frame counts.

    The cumulative sum is rounded only once, so rounding errors do not
    accumulate across units (phonemes, notes, ...) and the total frame count
    equals ``round(sum(dur_sec) / timestep)``.
    """
    dur_sec = torch.as_tensor(dur_sec, device=device)
    acc = torch.round(torch.cumsum(dur_sec, dim=-1) / timestep).long()
    return acc - F.pad(acc, (1, -1))  # same as acc - [0, acc[0], ..., acc[-2]]


@torch.no_grad()
def get_mel2ph_torch(lr, dur_sec, length, timestep, device='cpu'):
    dur = dur_sec_to_frames(dur_sec, timestep, device=device)
    mel2ph = lr(dur[None])[0]
    num_frames = mel2ph.shape[0]
    if num_frames < length:
        mel2ph = torch.cat((mel2ph, torch.full((length - num_frames,), fill_value=mel2ph[-1], device=device)), dim=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph


def get_pitch_parselmouth(
        waveform, samplerate, length,
        *, hop_size, f0_min=65, f0_max=1100,
        speed=1, interp_uv=False
):
    """

    :param waveform: [T]
    :param samplerate: sampling rate
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param f0_min: Minimum f0 in Hz
    :param f0_max: Maximum f0 in Hz
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hop_size * speed))
    time_step = hop_size / samplerate

    l_pad = int(np.ceil(1.5 / f0_min * samplerate))
    r_pad = hop_size * ((len(waveform) - 1) // hop_size + 1) - len(waveform) + l_pad + 1
    waveform = np.pad(waveform, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(waveform, sampling_frequency=samplerate).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    )
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_energy_librosa(waveform, length, *, hop_size, win_size, domain='db'):
    """
    Definition of energy: RMS of the waveform, in dB representation
    :param waveform: [T]
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: db or amplitude
    :return: energy
    """
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    energy = energy[: length]
    if domain == 'db':
        energy = librosa.amplitude_to_db(energy, top_db=None)
    elif domain == 'amplitude':
        pass
    else:
        raise ValueError(f'Invalid domain: {domain}')
    return energy


def get_breathiness(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of breathiness: RMS of the aperiodic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: breathiness
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_ap = waveform.aperiodic()
    breathiness = get_energy_librosa(
        waveform_ap, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return breathiness


def get_voicing(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of voicing: RMS of the harmonic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: voicing
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_sp = waveform.harmonic()
    voicing = get_energy_librosa(
        waveform_sp, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return voicing


def get_tension_base_harmonic(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None,
        domain='logit'
):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic)
    to the full harmonic part.
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_h = waveform.harmonic()
    waveform_base_h = waveform.harmonic(0)
    energy_base_h = get_energy_librosa(
        waveform_base_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
        domain='amplitude'
    )
    energy_h = get_energy_librosa(
        waveform_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
        domain='amplitude'
    )
    tension = np.sqrt(np.clip(energy_h ** 2 - energy_base_h ** 2, a_min=0, a_max=None)) / (energy_h + 1e-5)
    if domain == 'ratio':
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == 'db':
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension, top_db=None)
    elif domain == 'logit':
        tension = np.clip(tension, a_min=1e-4, a_max=1 - 1e-4)
        tension = np.log(tension / (1 - tension))
    return tension


class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=max(kernel_size, 1),
            bias=False,
            padding='same',
            padding_mode='replicate'
        )
        if kernel_size > 1:
            smooth_kernel = torch.sin(torch.from_numpy(
                np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi
            ))
            smooth_kernel /= smooth_kernel.sum()
        else:
            smooth_kernel = torch.tensor([1.0], dtype=torch.float32)
        self.weight.data = smooth_kernel[None, None]


# Kept at module level instead of on the binarizer instances: pickling torch tensors to
# worker processes goes through shared memory, which fails on Windows.
@lru_cache(maxsize=8)
def _get_smoothing_conv(kernel_size: int, device: torch.device) -> SinusoidalSmoothingConv1d:
    """Build and cache the sinusoidal smoothing convolution."""
    return SinusoidalSmoothingConv1d(kernel_size).eval().to(device)


def get_smoothing_conv(kernel_size: int, device=None) -> SinusoidalSmoothingConv1d:
    """Resolve the device before looking up the cached convolution."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _get_smoothing_conv(kernel_size, device)
