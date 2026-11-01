# -*- coding: utf-8 -*-
"""Degradation-aware modules for the Stage-2 experiments.

Insertion point: src/model/net.py, between the shallow feature extractors
(``feat_extractor_ir/vi``) and the alignment modules (``aam_ir/aam_vi``), i.e.
right after the per-frame features are computed and before any cross-frame
mixing.  Evidence for putting a module there: the whole degradation collapse is
attributable to the feature path (see day4_decomposition_report.md) -- replaying
a perfect clean optical flow recovers nothing.

Three variants are provided so the Stage-2 controls can be run from one code
path:

    DegradationEstimator + FilmedRectifier(use_cond=True)   -> DAR (predicted)
    DegradationEstimator bypassed, oracle descriptor fed in -> DAR (oracle)
    FilmedRectifier(use_cond=False)                         -> no-conditioning
                                                               ablation
    InputRestorer applied to the raw frames                 -> input-side
                                                               denoising control

All rectifier output layers are zero-initialised, so at step 0 every variant is
an exact identity and the model reproduces the released behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationEstimator(nn.Module):
    """Predict a small degradation descriptor from the raw 5-frame input.

    The descriptor is deliberately low dimensional (noise level, brightness and
    their per-modality contrast) because the downstream rectifier only needs
    enough information to select a correction regime -- a large descriptor would
    make the module hard to attribute in the ablation.
    """

    def __init__(self, in_ch: int = 3, cond_dim: int = 5, hidden: int = 32,
                 use_hf: bool = True, cross_modal: bool = False,
                 spectral: bool = False, nostride: bool = False,
                 temporal: bool = False):
        """``use_hf`` and ``cross_modal`` fix two measured blind spots.

        The original stem downsamples 4x and then globally pools, so it is
        *structurally incapable* of seeing high-frequency content -- which is why
        the deployable module recovers nothing on blur (+0.19%) while an oracle
        descriptor recovers 43.4%.  ``use_hf`` adds a local-contrast channel
        computed before any downsampling.

        ``cross_modal`` feeds both modalities to one estimator, giving it the
        infra-red/visible relationship.  Low-light applied to a single modality is
        ambiguous (a darkened bright scene and a genuinely dark scene look the
        same), but the cross-modal ratio breaks that tie.
        """
        super().__init__()
        self.cross_modal = cross_modal
        self.use_hf = use_hf
        # Blur is a SPECTRAL phenomenon: it removes high-frequency energy.  The
        # stem cannot see it because it downsamples 4x and then pools globally, so
        # the evidence is destroyed before it reaches the descriptor.  ``spectral``
        # appends hand-computed multi-scale energy statistics that are pooled
        # BEFORE any downsampling, which makes blur identification a property of
        # the representation rather than something the conv stem must discover.
        self.spectral = spectral
        self.n_spec = 9 if spectral else 0
        # Control for "you just did feature engineering".  ``nostride`` keeps the
        # same convolutional stem but removes every stride, so the learned
        # representation retains high frequencies all the way to the pooling.  If
        # the claim that downsampling destroys the blur evidence is right, a
        # no-stride stem should identify blur on its own, without the hand-computed
        # statistics; if it still cannot, the spectral prior is doing real work
        # rather than saving a conv layer.
        self.nostride = nostride
        # Temporal degradation fingerprint.  Video carries a cue that no single
        # image has: degradations differ in how they behave ACROSS frames --
        # sensor noise is i.i.d. per frame, a blur is temporally smooth, a tone
        # mapping drifts slowly.  ``temporal`` appends the frame-to-frame energy
        # statistics that separate those cases, which is exactly the cue the
        # spatial stem and the spectral statistics were missing for blur.
        self.temporal = temporal
        self.n_temporal = 3 if temporal else 0
        ch = in_ch * (2 if cross_modal else 1) + ((2 if cross_modal else 1) if use_hf else 0)
        stride = 1 if nostride else 2
        self.stem = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(hidden + self.n_spec + self.n_temporal, hidden),
                                  nn.GELU(), nn.Linear(hidden, cond_dim))
        # Auxiliary degradation classifier on the descriptor.  Without an explicit
        # incentive the descriptor is free to stay non-discriminative (the rectifier
        # can compensate unconditionally), which is what made the first `predicted`
        # run much weaker than `oracle` on the trained degradations (dark: +3.3% vs
        # +58.8%).  The synthetic labels are known, so supervising the descriptor
        # directly costs nothing.
        self.cls = nn.Linear(cond_dim, 5)
        # Regression head on the actual degradation parameters (noise sigma, gamma).
        # More informative than a class label for the rectifier: it asks the
        # descriptor to encode HOW degraded the input is, not just which bucket.
        self.reg = nn.Linear(cond_dim, 2)
        # Degradation gate: a single logit saying whether a correction is needed at
        # all.  Real-video validation showed that the ungated module LOSES 3.5-4.1%
        # on clean VTMOT/HDO video, because the descriptor is out of domain there
        # and reports a degradation that is not present.  The module needs a way to
        # say "leave this frame alone", and the supervision for it is free: we know
        # which training steps were clean.
        self.gate_head = nn.Linear(cond_dim, 1)
        # Prototype (vector-quantisation) code.  Measured motivation: a one-hot
        # oracle descriptor lets the rectifier learn a blur-specific correction
        # worth +43.4%, while a CONTINUOUS descriptor carrying the same information
        # (probe 0.873, raw 0.953) recovers +0.4%.  Information is not the
        # bottleneck -- the FORM of the conditioning is.  Emitting a soft
        # assignment over K prototypes gives the rectifier an almost-discrete
        # condition while keeping the estimator differentiable.
        self.n_proto = 0
        self.proto_head = None

    @staticmethod
    def _spectral_stats(x):                     # x: [B*T, C, H, W] -> [B*T, 9]
        """Per-level high-frequency energy, pooled at each level's own resolution.

        Level l energy = |I_l - upsample(downsample(I_l))|; mean, max and std of
        that map.  Blur flattens exactly these numbers, so a linear probe on them
        can name the degradation that the conv stem cannot.
        """
        cur = x.mean(dim=1, keepdim=True)
        out = []
        for _ in range(3):
            low = F.avg_pool2d(cur, kernel_size=2)
            low = F.interpolate(low, size=cur.shape[-2:], mode="bilinear", align_corners=False)
            hf = (cur - low).abs()
            out += [hf.mean(dim=(2, 3)), hf.amax(dim=(2, 3)), hf.std(dim=(2, 3))]
            cur = low
        return torch.cat(out, dim=1)

    @staticmethod
    def _temporal_stats(x):                     # x: [B, T, C, H, W] -> [B, 3]
        """Frame-to-frame energy statistics: the video-only degradation cue.

        Sensor noise is independent between frames, so the frame difference is
        large relative to the spatial high-frequency energy.  A blur is
        temporally smooth, so that ratio is small even though the difference
        itself is not.  A tone mapping changes the global level slowly, so both
        are small.  These three numbers therefore separate degradations that a
        single frame cannot.
        """
        b, t, c, h, w = x.shape
        xx = x.reshape(b * t, c, h, w)
        hf = DegradationEstimator._spectral_stats(xx).reshape(b, t, -1)[:, :, 0]  # level-1 hf mean
        cur = x.mean(dim=2)                                                   # [B, T, H, W]
        d = (cur[:, 1:] - cur[:, :-1]).abs().mean(dim=(2, 3))                  # [B, T-1]
        ratio = d / (hf[:, 1:] + 1e-6)
        return torch.stack([d.mean(dim=1), d.std(dim=1), ratio.mean(dim=1)], dim=1)

    @staticmethod
    def _high_frequency(x):                     # x: [B*T, C, H, W] -> [B*T, 1, H, W]
        c = x.mean(dim=1, keepdim=True)
        return (c - F.avg_pool2d(c, kernel_size=3, stride=1, padding=1)).abs()

    def attach_prototypes(self, n_proto: int, cond_dim: int):
        self.n_proto = n_proto
        self.proto_head = nn.Linear(cond_dim, n_proto)

    def code(self, c):
        """Map a descriptor to a (soft) prototype assignment of width K."""
        if self.proto_head is None:
            return c
        return torch.softmax(self.proto_head(c), dim=1)

    def forward(self, x, other=None):           # x, other: [B, 5, 3, H, W]
        b, t, c, h, w = x.shape
        parts = []
        if self.cross_modal:
            parts.append(x)
            parts.append(other if other is not None else torch.zeros_like(x))
        else:
            parts.append(x)
        xx = torch.cat(parts, dim=2).reshape(b * t, -1, h, w)
        if self.use_hf:
            if self.cross_modal:
                hf = [self._high_frequency(p.reshape(b * t, -1, h, w)) for p in parts]
                xx = torch.cat([xx] + hf, dim=1)
            else:
                xx = torch.cat([xx, self._high_frequency(xx)], dim=1)
        g = self.stem(xx).reshape(b, t, -1).mean(dim=1)
        if self.spectral:
            spec = self._spectral_stats(xx).reshape(b, t, -1).mean(dim=1)
            g = torch.cat([g, spec], dim=1)
        if self.temporal:
            g = torch.cat([g, self._temporal_stats(x)], dim=1)
        return self.head(g)                     # [B, cond_dim]


class FilmedRectifier(nn.Module):
    """FiLM-conditioned residual rectifier on a feature map.

    ``use_cond=False`` removes the conditioning path entirely, which is the
    ablation that shows whether conditioning itself contributes (as opposed to
    merely adding capacity).
    """

    def __init__(self, dim: int = 32, cond_dim: int = 5, hidden: int = 48,
                 n_blocks: int = 2, use_cond: bool = True):
        super().__init__()
        self.use_cond = use_cond
        self.inp = nn.Conv2d(dim, hidden, 3, 1, 1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GroupNorm(4, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1),
            ))
        if use_cond:
            self.film = nn.Sequential(nn.Linear(cond_dim, hidden), nn.GELU(),
                                      nn.Linear(hidden, 2 * hidden))
        self.out = nn.Conv2d(hidden, dim, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond=None, gate=None):  # x: [B, C, H, W]
        h = self.inp(x)
        if self.use_cond and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for blk in self.blocks:
            h = h + blk(h)
        residual = self.out(h)                  # zero-init out => identity at step 0
        if gate is not None:
            residual = residual * gate            # gate in [0,1]: 0 = leave the frame alone
        return x + residual


class InputRestorer(nn.Module):
    """Control: restore the raw frames instead of the features.

    This is the 'just denoise the input first' baseline.  It exists to test
    whether feature-side conditional rectification is actually better than
    cleaning the image before the network sees it.
    """

    def __init__(self, ch: int = 3, hidden: int = 32, n_blocks: int = 2):
        super().__init__()
        self.inp = nn.Conv2d(ch, hidden, 3, 1, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
                          nn.Conv2d(hidden, hidden, 3, 1, 1))
            for _ in range(n_blocks)
        ])
        self.out = nn.Conv2d(hidden, ch, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):                       # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        hh = self.inp(x.reshape(b * t, c, h, w))
        for blk in self.blocks:
            hh = hh + blk(hh)
        y = self.out(hh).reshape(b, t, c, h, w)
        return (x + y).clamp(0, 1)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())
