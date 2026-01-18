"""
Adapted from https://github.com/duchenzhuang/FSQ-pytorch/blob/main/quantizers/fsq.py#L41
"""
import torch
from torch import nn
from einops import rearrange


class FSQSTE(nn.Module):
    def __init__(self, levels):
        super().__init__()
        if levels:
            self.dim = len(levels)
            
            # Use register_buffer so these move to GPU automatically
            levels_tensor = torch.tensor(levels, dtype=torch.int32).view(1, 1, self.dim)
            self.register_buffer("levels", levels_tensor)
            
            half_levels = (self.levels - 1) * (1 - 1e-3) / 2
            self.register_buffer("half_levels", half_levels)
            
            offset = 0.5 - 0.5 * (self.levels % 2)
            self.register_buffer("offset", offset)
            
            shift = torch.tan(self.offset / self.half_levels)
            self.register_buffer("shift", shift)
            
            basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
            self.register_buffer("_basis", basis)
        else:
            self.levels = None

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self.levels
        return codes_non_centered

    def to_codebook_index(self, zhat):
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        indices = (zhat * self._basis).sum(dim=-1).round().to(torch.int32)
        return indices
    
    def from_codebook_index(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def forward(self, x):
        if self.levels is not None:
            x = torch.tanh(x + self.shift) * self.half_levels - self.offset
            x = x + (x.round() - x).detach()
            x = x / (self.levels // 2)
        return x