"""Pre-registered motion-score tie breakers for PR1."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


VALID_MODES = {"legacy", "tie_aux", "tie_var", "tie_rand"}


class PatchSelector:
    """Select a fixed Top-K patch budget without changing motion scores."""

    def __init__(self, mode: str = "legacy", seed: int = 0):
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown selector mode: {mode}")
        self.mode = mode
        self.seed = int(seed)

    @staticmethod
    def _patch_map(value: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
        if value.shape[-2:] != (height, width):
            value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size
        value = F.pad(value, (0, padded_width - width, 0, padded_height - height))
        return F.adaptive_avg_pool2d(value, (padded_height // patch_size, padded_width // patch_size)).view(-1)

    @staticmethod
    def _patch_variance(features: torch.Tensor, patch_size: int) -> torch.Tensor:
        _, _, height, width = features.shape
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size
        padded = F.pad(features, (0, padded_width - width, 0, padded_height - height))
        patches = padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches.var(dim=(-1, -2), unbiased=False).mean(dim=1).reshape(-1)

    @staticmethod
    def _tie_break(scores: torch.Tensor, secondary: torch.Tensor, k: int) -> torch.Tensor:
        threshold = torch.topk(scores, k, sorted=True).values[-1]
        strict = torch.nonzero(scores > threshold, as_tuple=False).flatten()
        boundary = torch.nonzero(scores == threshold, as_tuple=False).flatten()
        needed = k - strict.numel()
        boundary_order = torch.argsort(secondary.index_select(0, boundary), descending=True, stable=True)
        return torch.cat([strict, boundary[boundary_order[:needed]]], dim=0)

    def select(
        self,
        scores: torch.Tensor,
        k: int,
        patch_size: int,
        aux_map: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected indices and corresponding original motion scores."""
        if self.mode == "legacy":
            values, indices = torch.topk(scores, k, sorted=False)
            return indices, values

        if self.mode == "tie_aux":
            if aux_map is None or features is None:
                raise ValueError("tie_aux requires aux_map and features")
            aux = self._patch_map(aux_map, features.shape[-2], features.shape[-1], patch_size)
            indices = self._tie_break(scores, aux, k)
            return indices, scores.index_select(0, indices)

        if self.mode == "tie_var":
            if features is None:
                raise ValueError("tie_var requires features")
            variance = self._patch_variance(features, patch_size)
            indices = self._tie_break(scores, variance, k)
            return indices, scores.index_select(0, indices)

        threshold = torch.topk(scores, k, sorted=True).values[-1]
        strict = torch.nonzero(scores > threshold, as_tuple=False).flatten()
        boundary = torch.nonzero(scores == threshold, as_tuple=False).flatten()
        needed = k - strict.numel()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        permutation = torch.randperm(boundary.numel(), generator=generator)
        chosen_boundary = boundary[permutation[:needed].to(boundary.device)]
        indices = torch.cat([strict, chosen_boundary], dim=0)
        return indices, scores.index_select(0, indices)
