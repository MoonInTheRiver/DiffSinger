"""Word-level helpers for duration prediction."""

import torch
from torch import Tensor

__all__ = [
    "word_membership",
    "word_sizes",
    "word_position_ids",
    "word_distribution",
    "word_log_softmax",
    "word_budget",
]

# Large negative logit used instead of -inf so that empty rows stay finite.
_MASKED_LOGIT = -1e4
_EPS = 1e-8


def word_membership(word_ids: Tensor, num_words=None, x_masks: Tensor = None) -> Tensor:
    """One-hot membership mask of the words.

    :param word_ids: [B, T] word index per item, 1-based, 0 for padding
    :param num_words: number of words; defaults to the largest word index
    :param x_masks: [B, T] bool mask, True for padded items
    :return: [B, T, W] bool mask
    """
    if num_words is None:
        num_words = word_ids.max()
    index = torch.arange(1, num_words + 1, device=word_ids.device)
    mask = word_ids[:, :, None] == index[None, None, :]
    if x_masks is not None:
        mask = mask & ~x_masks.bool()[:, :, None]
    return mask


def word_sizes(mask: Tensor) -> Tensor:
    """:param mask: [B, T, W] membership mask; :return: [B, W] items per word"""
    return mask.sum(dim=1)


def word_position_ids(mask: Tensor, max_position: int = None):
    """Positions of every item inside its word, counted forward and backward.

    :param mask: [B, T, W] membership mask
    :param max_position: optional upper bound for position indices
    :return: (forward, reverse) [B, T] long tensors; padded items map to 0
    """
    numeric = mask.to(torch.int64)  # CumSum has no boolean kernel
    running = numeric.cumsum(dim=1) * numeric  # [B, T, W]
    forward = running.sum(dim=2) - 1  # [B, T]
    size = (word_sizes(mask)[:, None, :] * numeric).sum(dim=2)  # [B, T]
    reverse = size - 1 - forward
    forward = forward.clamp_min(0)
    reverse = reverse.clamp_min(0)
    if max_position is not None:
        forward = forward.clamp_max(max_position)
        reverse = reverse.clamp_max(max_position)
    return forward, reverse


def word_distribution(duration: Tensor, word_ids: Tensor, mask: Tensor = None) -> Tensor:
    """Normalize durations into a distribution over the phonemes of each word.

    :param duration: [B, T] durations or scores
    :param word_ids: [B, T] word index per item, 0 for padding
    :param mask: optional [B, T] bool mask of items to include; by default every
        item with a positive word index is included
    :return: [B, T] share of each item inside its word, zero elsewhere
    """
    if mask is None:
        mask = word_ids > 0
    shape = duration.shape[0], word_ids.max() + 1
    total = duration.new_zeros(*shape).scatter_add(
        1, word_ids, duration * mask
    )[:, 1:]  # [B, T] => [B, W]
    per_word = total.gather(1, word_ids.clamp(min=1) - 1)  # word total per item
    return duration * mask / per_word.clamp_min(_EPS)


def word_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log softmax of ``logits`` normalized within each word.

    :param logits: [B, T] unnormalized scores
    :param mask: [B, T, W] membership mask, padded items must be unset
    :return: [B, T] log share of each item inside its word
    """
    expanded = logits[:, :, None].masked_fill(~mask, _MASKED_LOGIT)  # [B, T, W]
    word_max = expanded.max(dim=1).values  # [B, W]
    shifted = (expanded - word_max[:, None, :]).exp() * mask
    log_norm = word_max + shifted.sum(dim=1).clamp_min(_EPS).log()  # [B, W]
    return ((expanded - log_norm[:, None, :]) * mask).sum(dim=2)  # [B, T]


def word_budget(durations: Tensor, word_ids: Tensor, num_words=None) -> Tensor:
    """Sum phoneme durations per word.

    :param durations: [B, T] per-item durations in frames
    :param word_ids: [B, T] word index per item, 1-based, 0 for padding
    :param num_words: number of words; defaults to the largest word index
    :return: [B, W] frame budget of every word
    """
    if num_words is None:
        num_words = word_ids.max()
    shape = durations.shape[0], num_words + 1
    return durations.new_zeros(*shape).scatter_add(1, word_ids, durations)[:, 1:]
