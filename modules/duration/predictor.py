"""Duration predictor built from sliding-window attention blocks."""

import torch
import torch.nn as nn

from modules.duration.attention import SlidingWindowBlock
from modules.duration.word_groups import (
    word_log_softmax,
    word_membership,
    word_position_ids,
)

__all__ = ["DurationRatioPredictor"]


class DurationRatioPredictor(nn.Module):
    """Distribute each word's frame budget with sliding-window attention.

    ``use_pos_embed`` adds within-word positions, clamped to ``radius``.
    """

    def __init__(self, in_dims, hidden_size, num_blocks=4, num_heads=4, radius=8,
                 ffn_mult=4, ffn_act='gelu', dropout=0.0, use_pos_embed=True):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_pos_embed = use_pos_embed
        self.max_pos = radius
        self.in_proj = nn.Linear(in_dims, hidden_size)
        self.blocks = nn.ModuleList(
            SlidingWindowBlock(
                hidden_size, num_heads, radius, ffn_mult, ffn_act, dropout
            )
            for _ in range(num_blocks)
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)
        if use_pos_embed:
            self.pos_embed = nn.Embedding(self.max_pos + 1, hidden_size)
            self.reverse_pos_embed = nn.Embedding(self.max_pos + 1, hidden_size)
        else:
            self.pos_embed = None
            self.reverse_pos_embed = None

    @classmethod
    def from_hparams(cls, in_dims, dur_hparams: dict) -> 'DurationRatioPredictor':
        radius = dur_hparams.get('dur_radius', 8)
        return cls(
            in_dims=in_dims,
            hidden_size=dur_hparams['hidden_size'],
            num_blocks=dur_hparams.get('dur_num_blocks', 4),
            num_heads=dur_hparams.get('dur_num_heads', 4),
            radius=radius,
            ffn_mult=dur_hparams.get('dur_ffn_mult', 4),
            dropout=dur_hparams['dropout'],
        )

    def forward(self, xs, x_masks=None, ph2word=None, word_budget=None):
        """Predict per-phoneme frame counts.

        Args:
            xs: Input sequences (B, Tmax, idim).
            x_masks: Padding mask (B, Tmax).
            ph2word: Word index per phoneme [B, Tmax], 0 for padding.
            word_budget: Frame budget of every word [B, T_w].

        Returns:
            Continuous frame counts [B, Tmax]; downstream consumers handle
            rounding and word-boundary alignment.
        """
        if x_masks is None:
            non_pad_mask = torch.ones(xs.shape[:2], dtype=torch.bool, device=xs.device)
        else:
            non_pad_mask = ~x_masks.bool()
        if ph2word is None:
            raise ValueError(
                'the duration predictor needs the word ids of the phonemes; pass '
                'ph2word (training) or pass word_div to rebuild them (inference and export)'
            )
        if word_budget is None:
            raise ValueError(
                'the duration predictor needs the frame budget of every word; pass a '
                '[B, T_w] word_budget (training) or word_dur (inference and export)'
            )
        members = word_membership(ph2word, x_masks=x_masks)

        hidden = self.in_proj(xs)
        if self.pos_embed is not None:
            forward_pos, reverse_pos = word_position_ids(members, self.max_pos)
            hidden = hidden + self.pos_embed(forward_pos) + self.reverse_pos_embed(reverse_pos)
        hidden = hidden * non_pad_mask[:, :, None]
        for block in self.blocks:
            hidden = block(hidden, non_pad_mask)
        logits = self.out_proj(self.out_norm(hidden)).squeeze(-1)  # [B, Tmax]
        logits = logits * non_pad_mask

        # Use float32 to limit rounding errors under mixed precision.
        logits = logits.float()
        prob = word_log_softmax(logits, members).exp() * non_pad_mask
        budget = word_budget.gather(1, ph2word.clamp(min=1) - 1)
        return prob * budget
