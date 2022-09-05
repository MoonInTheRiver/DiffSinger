import numpy as np


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(result[:idx], a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = k * a[idx:] + (1 - k) * b[: fade_len]
    np.copyto(b[fade_len:], result[a.shape[0]:])
    return result
