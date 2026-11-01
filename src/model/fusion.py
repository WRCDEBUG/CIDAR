# MAVFusion: MDIM (Motion-Guided Dual-Interaction Module) and its building blocks
# Static branch (StaticUNetBlock, LightEncoder) handles weak interaction in static regions;
# dynamic branch (MultiScaleSparseBranch, MaskedDATBlock) performs Top-K sparse attention
# in dynamic regions identified by the motion mask.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .selector import PatchSelector


# ---------------------------------------------------------------------------
# Static branch: StaticUNetBlock (lightweight U-Net for local texture modeling)
# ---------------------------------------------------------------------------
class StaticUNetBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=2, bias=False):
        super().__init__()
        exp_dim = int(dim * expansion_ratio)

        # Encoder
        self.enc_conv1 = nn.Conv2d(dim, exp_dim, kernel_size=3, padding=1, bias=bias)
        self.enc_dw = nn.Conv2d(exp_dim, exp_dim, kernel_size=3, padding=1, groups=exp_dim, bias=bias)
        self.enc_norm = nn.GroupNorm(4, exp_dim)

        # Bottleneck (Scale-Space)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(exp_dim, exp_dim, kernel_size=5, padding=2, groups=exp_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(exp_dim, exp_dim, kernel_size=3, padding=1, groups=exp_dim, bias=bias)
        )

        # Decoder
        self.dec_pw = nn.Conv2d(exp_dim, exp_dim, kernel_size=1, bias=bias)
        self.dec_conv2 = nn.Conv2d(exp_dim, dim, kernel_size=3, padding=1, bias=bias)

        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, x):
        identity = x
        e1 = self.act(self.enc_norm(self.enc_dw(self.enc_conv1(x))))
        low_res = F.avg_pool2d(e1, kernel_size=2, stride=2)
        b = self.bottleneck(low_res)
        up = F.interpolate(b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        fusion = self.act(self.dec_pw(e1 + up))
        out = self.dec_conv2(fusion)
        return identity + self.gamma * out


class MaskedDATBlock(nn.Module):
    def __init__(self, dim, num_heads=4, topk=32, ffn_expansion=2.66, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.topk = topk
        self.scale = (dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)

        hidden_dim = int(dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias)
        )

    def forward(self, x, global_tokens):
        Bp, C, p1, p2 = x.shape
        x_token = x.flatten(2).mean(-1)
        residual = x_token
        x_norm = self.norm1(x_token)

        # Joint QKV: query from active patch, key/value from global tokens
        qkv = self.qkv(torch.cat([x_norm, global_tokens], dim=0))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q[:Bp]
        k, v = k[Bp:], v[Bp:]

        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'm (h d) -> h m d', h=self.num_heads)
        v = rearrange(v, 'm (h d) -> h m d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        tk = min(self.topk, attn.shape[-1])
        val, idx = attn.topk(tk, dim=-1)
        attn_sparse = torch.zeros_like(attn).scatter_(-1, idx, val).softmax(dim=-1)

        out = rearrange(attn_sparse @ v, 'h n d -> n (h d)')
        x_token = residual + self.proj(out)
        x_token = x_token + self.ffn(self.norm2(x_token))
        return x_token.view(Bp, C, 1, 1).expand(-1, -1, p1, p2)


class MultiScaleSparseBranch(nn.Module):
    def __init__(self, dim, num_blocks, head=4, patch_size=8, tau=0.3, topk_ratio=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.tau, self.topk_ratio = tau, topk_ratio

        self.compress = nn.Conv2d(dim, dim, 1, bias=False)
        self.excavate = nn.Conv2d(dim, dim, 1, bias=False)

        self.blocks = nn.ModuleList([
            MaskedDATBlock(dim, num_heads=head, topk=64)
            for _ in range(num_blocks)
        ])

        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.layer_name = None
        self.selector = None
        # "fixed"   : original behaviour, k = max(1, int(topk_ratio * N)).
        # "support" : k = min(fixed_k, A), A = number of patches with motion
        #             score > 0.  Selected zero-score patches are multiplied by
        #             val = scores[idx] = 0 and contribute exactly nothing, so
        #             this is equivalence-preserving.  See
        #             runs/00_contract/step0_report.md.
        self.budget_mode = "fixed"
        # --- decoupling / allocation switches (defaults reproduce the baseline) ---
        # score_mode : "motion"           -> shipped
        #              "motion_saliency"  -> motion + w * local-contrast saliency (fills the
        #                                    leftover budget with static-but-salient patches)
        #              "saliency"         -> saliency only
        #              "random"           -> random allocation (cheap control)
        # amp_mode   : "released" -> soft_weights = val (motion score)
        #              "unity"    -> soft_weights = 1  (decoupled amplitude)
        self.score_mode = "motion"
        self.sal_weight = 0.0
        self.amp_mode = "released"
        # Audit instrumentation.  Off by default: it retains every layer's
        # activations, which in a production inference loop grows without bound
        # and distorts latency measurements.  Audit tools call set_recording(True).
        self.record_enabled = False
        self.selection_records = []
        self.gate_records = []
        # cheap integer counters, always on: no tensor retention, no sync
        self.k_total = 0
        self.support_total = 0
        self.zero_support_calls = 0

    def configure_selector(self, layer_name, selector=None):
        self.layer_name = layer_name
        self.selector = selector

    def clear_records(self):
        self.selection_records.clear()
        self.gate_records.clear()

    def reset_budget_stats(self):
        self.k_total = 0
        self.support_total = 0
        self.zero_support_calls = 0

    def budget_stats(self):
        return {"k_total": self.k_total, "support_total": self.support_total,
                "zero_support_calls": self.zero_support_calls}

    def forward(self, x, mask, aux_map=None):
        B, C, H, W = x.shape
        p = self.patch_size
        if isinstance(p, (list, tuple)):
            p = p[0]
        x_comp = self.compress(x)

        # Gradient sink prevents DDP "unused parameters" warnings by
        # forcing a 0-coefficient connection to all block parameters.
        dummy_grad_sink = sum(p.view(-1)[0] for p in self.blocks.parameters()) * 0.0

        # Pad to a multiple of patch size
        if H < p or W < p:
            return torch.zeros_like(x_comp) + dummy_grad_sink

        hp, wp = ((H + p - 1) // p) * p, ((W + p - 1) // p) * p
        m_pad = F.pad(mask, (0, wp - W, 0, hp - H))

        # Per-patch saliency from motion mask
        scores = F.adaptive_avg_pool2d(m_pad, (hp // p, wp // p)).view(-1)

        # Allocation score.  "motion" reproduces the shipped behaviour exactly.
        if self.score_mode == "motion":
            select_scores = scores
        else:
            c = x.mean(dim=1, keepdim=True)
            local = (c - F.avg_pool2d(c, kernel_size=9, stride=1, padding=4)).abs()
            local = F.pad(local, (0, wp - W, 0, hp - H))
            sal = F.adaptive_avg_pool2d(local, (hp // p, wp // p)).view(-1)
            sal = sal / (sal.max() + 1e-6)
            if self.score_mode == "motion_saliency":
                select_scores = scores + self.sal_weight * sal
            elif self.score_mode == "saliency":
                select_scores = sal
            else:
                select_scores = torch.rand_like(scores)
        k = max(1, int(self.topk_ratio * scores.numel()))
        support = None
        if self.budget_mode == "support":
            # A zero-score patch is weighted by val = 0 downstream, so it can
            # never change the output.  Cap the budget at the motion support.
            support = int((scores > 0).sum().item())
            k = max(1, min(k, support))
        if self.selector is None:
            val, active_idx = torch.topk(select_scores, k, sorted=False)
            if self.amp_mode == "unity":
                val = torch.ones_like(val)
        else:
            active_idx, val = self.selector.select(
                scores=scores,
                k=k,
                patch_size=p,
                aux_map=aux_map,
                features=x,
            )

        self.k_total += int(k)
        if support is not None:
            self.support_total += int(support)
            self.zero_support_calls += int(support == 0)

        if self.record_enabled:
            self.selection_records.append({
            "layer": self.layer_name,
            "scores": scores.detach(),
            "active_idx": active_idx.detach(),
            "k": k,
            "support": support,
            "budget_mode": self.budget_mode,
            "shape": (hp, wp),
            })

        # Split into patches; global context = mean over spatial dims
        x_pad = F.pad(x_comp, (0, wp - W, 0, hp - H))
        x_p = rearrange(x_pad, 'b c (nh p1) (nw p2) -> (b nh nw) c p1 p2', p1=p, p2=p)
        g_tokens = x_p.mean(dim=(2, 3))
        # ECR_ROUTE_HOOK_V1: retain all original branch arithmetic and amplitudes.
        if getattr(self, "exchange_router", None) is not None:
            active_idx = self.exchange_router(
                g_tokens, scores, active_idx,
                batch_size=B, grid_hw=(hp // p, wp // p),
            )
            val = scores.index_select(0, active_idx)

        # Every selected patch has score 0: the whole dynamic branch reduces to
        # proj_out(0), so the attention can be skipped entirely.
        if support == 0:
            empty = torch.zeros_like(x_comp)
            return self.proj_out(empty + dummy_grad_sink)

        # Apply attention only to active patches
        act_p = self.compress(x_p[active_idx])
        for blk in self.blocks:
            act_p = blk(act_p, g_tokens)

        # Scatter back with saliency-weighted contribution
        soft_weights = val.view(-1, 1, 1, 1)
        act_p = self.excavate(act_p) * soft_weights

        diff = torch.zeros_like(x_p)
        diff.index_copy_(0, active_idx.long(), act_p)

        diff_map = rearrange(diff, '(b nh nw) c p1 p2 -> b c (nh p1) (nw p2)',
                             b=B, nh=hp // p, nw=wp // p)[:, :, :H, :W]

        return self.proj_out(diff_map + dummy_grad_sink)


class HybridBlock(nn.Module):
    def __init__(self, dim, num_blocks=1, head=4, patch_sizes=(8, 16)):
        super().__init__()
        self.static_branch = StaticUNetBlock(dim)
        self.dynamic_branch = MultiScaleSparseBranch(dim, num_blocks, head, patch_sizes)
        self.smooth = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.layer_name = None
        # "released" -> out = s_feat + d_feat * m_s   (pixel motion gate)
        # "none"     -> out = s_feat + d_feat         (decoupled pixel gate)
        self.gate_mode = "released"

    def configure_selector(self, layer_name, selector=None):
        self.layer_name = layer_name
        self.dynamic_branch.configure_selector(layer_name, selector)

    def configure_budget(self, mode="fixed"):
        self.dynamic_branch.budget_mode = mode

    def clear_records(self):
        self.dynamic_branch.clear_records()

    def forward(self, x, mask, aux_map=None):
        s_feat = self.static_branch(x)
        d_feat = self.dynamic_branch(x, mask, aux_map=aux_map)

        m_s = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        gated_dynamic = d_feat if self.gate_mode == "none" else d_feat * m_s
        out = s_feat + gated_dynamic
        smooth_out = self.smooth(out)
        if self.dynamic_branch.record_enabled:
            self.dynamic_branch.gate_records.append({
                "layer": self.layer_name,
                "dynamic": d_feat.detach(),
                "mask": m_s.detach(),
                "out": out.detach(),
                "smooth": smooth_out.detach(),
            })
        return out + smooth_out


class LightEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim),
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU()
        )
        self.smooth = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim)

    def forward(self, x):
        out = self.body(x)
        return out + self.smooth(out)


class Fusion_Net(nn.Module):
    def __init__(self, dim=64, num_blocks=1, head=4):
        super().__init__()

        # Input projection: 2 * (dim//8) -> dim//4
        self.en_ir = LightEncoder(dim, dim // 8)
        self.en_vi = LightEncoder(dim, dim // 8)
        curr = dim // 4

        # --- Encoder ---
        self.e1 = HybridBlock(curr, num_blocks, head)
        self.d1 = nn.Conv2d(curr, curr * 2, 3, 2, 1)

        self.e2 = HybridBlock(curr * 2, num_blocks, head)
        self.d2 = nn.Conv2d(curr * 2, curr * 4, 3, 2, 1)

        self.e3 = HybridBlock(curr * 4, num_blocks, head)
        self.d3 = nn.Conv2d(curr * 4, curr * 4, 3, 2, 1)

        # --- Bottleneck ---
        self.bt = HybridBlock(curr * 4, num_blocks, head)

        # --- Decoder (channel widths in comments denote 2x up-concat channel counts) ---
        self.up3 = nn.ConvTranspose2d(curr * 4, curr * 4, 2, 2)
        self.re3 = nn.Conv2d(curr * 8, curr * 4, 1)
        self.dec3 = HybridBlock(curr * 4, num_blocks, head)

        self.up2 = nn.ConvTranspose2d(curr * 4, curr * 2, 2, 2)
        self.re2 = nn.Conv2d(curr * 4, curr * 2, 1)
        self.dec2 = HybridBlock(curr * 2, num_blocks, head)

        self.up1 = nn.ConvTranspose2d(curr * 2, curr, 2, 2)
        self.re1 = nn.Conv2d(curr * 2, curr, 1)
        self.dec1 = HybridBlock(curr, num_blocks, head)

        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(curr, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.selector = None
        self.selector_layer = None
        self.budget_mode = "fixed"
        # --- decoupling / allocation switches (defaults reproduce the baseline) ---
        # score_mode : "motion"           -> shipped
        #              "motion_saliency"  -> motion + w * local-contrast saliency (fills the
        #                                    leftover budget with static-but-salient patches)
        #              "saliency"         -> saliency only
        #              "random"           -> random allocation (cheap control)
        # amp_mode   : "released" -> soft_weights = val (motion score)
        #              "unity"    -> soft_weights = 1  (decoupled amplitude)
        self.score_mode = "motion"
        self.sal_weight = 0.0
        self.amp_mode = "released"

    def configure_selector(self, mode="legacy", layer="dec1", seed=0):
        """Attach a PatchSelector to one layer, or to every layer when layer="all"."""
        if mode == "legacy":
            self.selector = PatchSelector(mode="legacy", seed=seed)
        else:
            self.selector = PatchSelector(mode=mode, seed=seed)
        self.selector_layer = layer
        layers = ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1") if layer == "all" else (layer,)
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            getattr(self, name).configure_selector(
                name, self.selector if name in layers else None
            )

    def clear_records(self):
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            getattr(self, name).clear_records()

    def configure_allocation(self, score_mode="motion", sal_weight=0.0,
                             amp_mode="released", gate_mode="released"):
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            blk = getattr(self, name)
            blk.gate_mode = gate_mode
            blk.dynamic_branch.score_mode = score_mode
            blk.dynamic_branch.sal_weight = sal_weight
            blk.dynamic_branch.amp_mode = amp_mode

    def set_recording(self, enabled=True):
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            getattr(self, name).dynamic_branch.record_enabled = bool(enabled)

    def reset_budget_stats(self):
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            getattr(self, name).dynamic_branch.reset_budget_stats()
        self.budget_mode = self.budget_mode

    def budget_stats(self):
        agg = {"k_total": 0, "support_total": 0, "zero_support_calls": 0}
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            for key, value in getattr(self, name).dynamic_branch.budget_stats().items():
                agg[key] += value
        return agg

    def configure_budget(self, mode="fixed"):
        """Set the dynamic-branch budget policy on every fusion layer.

        mode="fixed"   reproduces the original int(topk_ratio * N) budget.
        mode="support" caps it at the per-frame motion support A (equivalence
                       preserving; see runs/00_contract/step0_report.md).
        """
        if mode not in ("fixed", "support"):
            raise ValueError(f"Unknown budget mode: {mode}")
        self.budget_mode = mode
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            getattr(self, name).configure_budget(mode)

    def get_records(self):
        records = {}
        for name in ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1"):
            block = getattr(self, name)
            records[name] = {
                "selection": list(block.dynamic_branch.selection_records),
                "gate": list(block.dynamic_branch.gate_records),
            }
        return records

    def forward(self, ir, vi, mask):
        x = torch.cat([self.en_ir(ir), self.en_vi(vi)], dim=1)
        aux_map = (ir - vi).abs().mean(dim=1, keepdim=True)

        skip1 = self.e1(x, mask, aux_map=aux_map)
        x = self.d1(skip1)

        skip2 = self.e2(x, mask, aux_map=aux_map)
        x = self.d2(skip2)

        skip3 = self.e3(x, mask, aux_map=aux_map)
        x = self.d3(skip3)

        x = self.bt(x, mask, aux_map=aux_map)

        x = self.up3(x)
        x = self.dec3(self.re3(torch.cat([x, skip3], 1)), mask, aux_map=aux_map)

        x = self.up2(x)
        x = self.dec2(self.re2(torch.cat([x, skip2], 1)), mask, aux_map=aux_map)

        x = self.up1(x)
        x = self.dec1(self.re1(torch.cat([x, skip1], 1)), mask, aux_map=aux_map)

        return self.recon(x)
