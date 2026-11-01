# MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan
# ECCV 2026
#
# The RAFT inference wrapper (RAFT / RAFT_component) is adapted from SEA-RAFT
# (Princeton-VL, https://github.com/princeton-vl/SEA-RAFT) and the engineering
# scaffolding is derived from UniVF (Zixiang Zhao et al., NeurIPS 2025 Spotlight).

import torch
import torch.nn as nn
import torch.nn.functional as F
from .raft import RAFT
from .RAFT_component.raft_utils import load_ckpt
from .fusion import *
from .utils import load_args_from_json, flow_warp


class LightweightAAM(nn.Module):
    def __init__(self, dim, base_channels=32):
        super().__init__()
        self.dim = dim

        # Lightweight refinement encoder (DWConv + PWConv)
        mid_channels = base_channels
        self.compress = nn.Conv2d(dim * 4 + 4, mid_channels, 1)  # 1x1 channel reduction

        self.refine_ds_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 2, 3, 1, 1)  # final residual flow
        )

        # Lightweight temporal aggregation (softmax-weighted sum, replacing Conv3d)
        self.temp_weight = nn.Parameter(torch.ones(1, 1, 3, 1, 1))

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )

        # Static branch fallback
        self.static_refine = nn.Conv2d(dim, dim, 1) if dim > 16 else nn.Identity()
        # diagnostic only: when set to a list, per-call temporal stream diversity
        # is appended so the degeneracy of the three aggregated streams is visible
        self.stash = None
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, f_curr, f_prev, f_next, flow_prev, flow_next, motion_mask, f_anchor):
        # A: Initial warp
        f_prev_a = flow_warp(f_prev, flow_prev)
        f_next_a = flow_warp(f_next, flow_next)
        flow_init = (flow_prev + flow_next) * 0.5
        f_curr_a = flow_warp(f_curr, flow_init)

        # B: Flow refinement
        res_in = torch.cat([f_anchor, f_prev_a, f_curr_a, f_next_a, flow_prev, flow_next], dim=1)
        res_feat = self.compress(res_in)
        flow_res = self.refine_ds_conv(res_feat)

        f_curr_final = flow_warp(f_curr, flow_init + flow_res)

        # C: Temporal aggregation (softmax-weighted, memory-efficient)
        st_feat = torch.stack([f_prev_a, f_curr_final, f_next_a], dim=2)
        if getattr(self, "stash", None) is not None:
            self.stash.append({
                "pair_prev_curr": float((st_feat[:, :, 0] - st_feat[:, :, 1]).abs().mean()),
                "pair_curr_next": float((st_feat[:, :, 1] - st_feat[:, :, 2]).abs().mean()),
            })
        feat_aggregated = (st_feat * torch.softmax(self.temp_weight, dim=2)).sum(dim=2)

        # D: Spatial refinement + motion-mask gating
        feat_refined = self.spatial_refine(feat_aggregated)

        f_static = self.static_refine(f_curr)
        out = feat_refined * motion_mask + f_static * (1 - motion_mask)

        return self.act(out)


class VideoFusion(nn.Module):
    def __init__(self, model_config, dar: dict | None = None):
        """``dar`` builds the optional Stage-2 degradation-aware modules.

        Default None means the modules are not constructed at all, so the
        released checkpoint still loads strictly and every existing script is
        unaffected.  Recognised keys:
            mode        : "off" | "oracle" | "predicted" | "noconv"
            input_restore : bool, apply InputRestorer to the raw frames
        """
        super(VideoFusion, self).__init__()
        self.dar_cfg = dar or {}
        self.dar_mode = self.dar_cfg.get("mode", "off")
        self.dar_cond_dim = self.dar_cfg.get("cond_dim", 5)
        self.dar_cond_vec = None            # oracle descriptor, set by the caller
        # Demand-driven conditioning: when True, the rectifier is fed the TRUE
        # descriptor even though the estimator also runs.  Training then makes the
        # rectification loss sensitive to descriptor quality, which creates the
        # gradient that forces the estimator to keep the information that plain
        # end-to-end training erodes.
        self.dar_force_oracle = False
        # When True the rectifier's residual is scaled by a learned gate, so the
        # module can abstain on frames that need no correction.
        self.dar_gate = bool(self.dar_cfg.get("gate", False))
        self.dar_vq = int(self.dar_cfg.get("vq", 0))
        self.last_gate = None
        self.dar_ir = None
        self.dar_vi = None
        self.dar_ir2 = None          # second injection point: fusion-net input
        self.dar_vi2 = None
        self.dar_estimator = None
        self.input_restorer = None

        # 1. Optical flow network (SEA-RAFT, frozen during training)
        self.raft_args = load_args_from_json("config/module/spring-S.json")
        self.flow_net = RAFT(self.raft_args).eval()
        load_ckpt(self.flow_net, self.raft_args.path)

        # Freeze RAFT parameters
        for param in self.flow_net.parameters():
            param.requires_grad = False

        # Motion-compensation direction.
        #   "released"  : flow_prev = get_flow(vi_prev, vi_curr)        (as shipped)
        #   "corrected" : flow_prev = get_flow(vi_curr, vi_prev)
        # Empirically (tools/flow_direction_check.py) the field that actually
        # aligns prev onto curr is get_flow(curr, prev); the shipped order
        # displaces the neighbour away from the current frame by roughly twice
        # the motion.  Default keeps the shipped behaviour bit-for-bit.
        self.flow_direction = "released"
        # Optical-flow compute budget (F1 levers).
        #   flow_max_res : SEA-RAFT input cap; 256 as shipped
        #   flow_budget  : "full" = 4 get_flow calls per 5-frame window (shipped)
        #                  "half" = 2 calls, later iterations reuse the cached field
        self.flow_max_res = 256
        self.flow_budget = "full"
        # Reliability-weighted motion gating (default off = shipped behaviour).
        #   "none"  : shipped binary mask
        #   "grad"  : suppress mask where the flow field is high-frequency
        #             (cheap proxy for noise-induced spurious motion)
        #   "fb"    : suppress mask where forward/backward flow is inconsistent
        self.reliability_mode = "none"
        self.rel_alpha = 1.0
        self.rel_floor = 0.0

        dim = model_config['model']['dim']
        # Shallow feature extractor (3 -> dim) for both IR and VI modalities
        self.feat_extractor_ir = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 3, 1, 1)
        )
        self.feat_extractor_vi = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 3, 1, 1)
        )
        self.aam_ir = LightweightAAM(dim)
        self.aam_vi = LightweightAAM(dim)
        self.fusion_net = Fusion_Net(dim=model_config['model']['dim'])
        self._build_dar(dim)

    def configure_pr1_selector(self, mode="legacy", layer="dec1", seed=0):
        self.fusion_net.configure_selector(mode=mode, layer=layer, seed=seed)

    def configure_allocation(self, **kwargs):
        self.fusion_net.configure_allocation(**kwargs)

    def set_recording(self, enabled=True):
        self.fusion_net.set_recording(enabled)

    def reset_budget_stats(self):
        self.fusion_net.reset_budget_stats()

    def budget_stats(self):
        return self.fusion_net.budget_stats()

    def configure_budget(self, mode="fixed"):
        """Dynamic-branch budget policy for every fusion layer.

        "fixed"   original k = max(1, int(0.3 * N)).
        "support" k = min(k, A), A = patches with motion score > 0.
                  Zero-score selections contribute exactly nothing, so this is
                  equivalence preserving; see runs/00_contract/step0_report.md.
        """
        self.fusion_net.configure_budget(mode)

    def clear_pr1_records(self):
        self.fusion_net.clear_records()

    def get_pr1_records(self):
        return self.fusion_net.get_records()

    def _build_dar(self, dim):
        """Called at the end of __init__ with the resolved feature width."""
        from .dar import DegradationEstimator, FilmedRectifier, InputRestorer
        if self.dar_cfg.get("input_restore", False):
            self.input_restorer = InputRestorer(ch=3)
        if self.dar_mode == "off":
            return
        use_cond = self.dar_mode in ("oracle", "predicted")
        # with prototypes the rectifier is conditioned on a K-way assignment
        rect_cond_dim = self.dar_vq if self.dar_vq > 0 else self.dar_cond_dim
        self.dar_ir = FilmedRectifier(dim=dim, cond_dim=rect_cond_dim,
                                      use_cond=use_cond)
        self.dar_vi = FilmedRectifier(dim=dim, cond_dim=rect_cond_dim,
                                      use_cond=use_cond)
        if self.dar_cfg.get("second", False):
            # Second injection point, at the input of the fusion network.  The
            # blind fine-tune's only advantage over the first-stage rectifier was
            # low-light, and it wins there precisely because it adapts fusion_net;
            # this puts conditioned adaptation at that same location.  Kept narrow
            # (hidden 32, one block) so the module stays light.
            self.dar_ir2 = FilmedRectifier(dim=dim, cond_dim=self.dar_cond_dim,
                                           hidden=32, n_blocks=1, use_cond=use_cond)
            self.dar_vi2 = FilmedRectifier(dim=dim, cond_dim=self.dar_cond_dim,
                                           hidden=32, n_blocks=1, use_cond=use_cond)
        if self.dar_mode == "predicted":
            # est_mode: "base" = original stem, "hf" = + high-frequency channel,
            # "cross" = + high-frequency and both modalities in one estimator
            em = self.dar_cfg.get("est_mode", "base")
            # "crossspec" = cross-modal + spectral statistics; "spectral" = spectral
            # only, which is the ablation that isolates the spectral cue.
            self.dar_estimator = DegradationEstimator(
                in_ch=3, cond_dim=self.dar_cond_dim,
                use_hf=em in ("hf", "cross", "crossspec", "crosstemporal"),
                cross_modal=em in ("cross", "crossspec", "crosstemporal"),
                spectral=em in ("spectral", "crossspec", "crosstemporal"),
                nostride=(em == "nostride"),
                temporal=em in ("temporal", "crosstemporal"))
            if self.dar_vq > 0:
                self.dar_estimator.attach_prototypes(self.dar_vq, self.dar_cond_dim)

    def get_flow(self, img1, img2, max_res=None):
        """
        Efficient optical-flow inference (adaptive resolution).

        Downscales inputs to ``max_res`` on the longer side, runs SEA-RAFT, then
        upsamples the flow back to the original resolution (and rescales
        magnitudes by ``1/scale``).
        """
        if max_res is None:
            max_res = getattr(self, "flow_max_res", 256)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            b, c, h, w = img1.shape

            scale_factor = min(max_res / max(h, w), 1.0)
            if scale_factor < 1.0:
                new_size = (int(h * scale_factor), int(w * scale_factor))
                imgs = torch.cat([img1, img2], dim=0)
                imgs_resized = F.interpolate(imgs, size=new_size, mode='bilinear', align_corners=False)
                img1_resized, img2_resized = imgs_resized.chunk(2, dim=0)
            else:
                img1_resized, img2_resized = img1, img2

            # Run SEA-RAFT (expects [0, 255] range)
            flow_result = self.flow_net(img1_resized * 255.0, img2_resized * 255.0, test_mode=True)

            # Extract the predicted flow tensor
            if isinstance(flow_result, dict):
                flow = flow_result.get('final', list(flow_result.values())[-1])
            elif isinstance(flow_result, (list, tuple)):
                flow = flow_result[-1]
            else:
                flow = flow_result

            target_h, target_w = img1_resized.shape[-2:]
            if flow.shape[-2:] != (target_h, target_w):
                flow_h, flow_w = flow.shape[-2:]
                flow = F.interpolate(
                    flow, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
                flow[:, 0] *= target_w / flow_w
                flow[:, 1] *= target_h / flow_h

            # If we downscaled, upsample back and rescale magnitudes
            if scale_factor < 1.0:
                flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
                flow = flow * (1.0 / scale_factor)

            return flow

    def generate_motion_mask_from_flow(self, flow_prev, flow_next):
        """
        Build a binary motion mask from forward / backward optical flow magnitudes.

        The mask is True where either of the two flows has magnitude > 0.5 px
        (followed by 3x3 max-pool dilation) and serves as a physical prior for
        the MDIM module to allocate sparse attention to moving regions.
        """
        with torch.no_grad():
            mag_prev = torch.sqrt(flow_prev[:, 0] ** 2 + flow_prev[:, 1] ** 2 + 1e-6)
            mag_next = torch.sqrt(flow_next[:, 0] ** 2 + flow_next[:, 1] ** 2 + 1e-6)
            mag = torch.max(mag_prev, mag_next)

            mask = (mag > 0.5).float().unsqueeze(1)
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)  # 3x3 dilation

        return mask

    def pair_flow(self, source, target):
        """Return the field that warps `source` onto `target` under the configured convention."""
        if self.flow_direction == "corrected":
            return self.get_flow(target, source)
        return self.get_flow(source, target)

    @staticmethod
    def _flow_grad_mag(flow):
        dx = (flow[..., 1:, :] - flow[..., :-1, :]).abs()
        dy = (flow[..., :, 1:] - flow[..., :, :-1]).abs()
        dx = F.pad(dx, (0, 0, 0, 1))   # (H-1, W) -> (H, W)
        dy = F.pad(dy, (0, 1, 0, 0))   # (H, W-1) -> (H, W)
        return (dx + dy).mean(dim=1, keepdim=True)

    def flow_reliability(self, flow_prev, flow_next, vi_prev, vi_curr, vi_next):
        """Per-pixel reliability in [0,1]; 1 = trustworthy motion."""
        if self.reliability_mode == "grad":
            err = 0.5 * (self._flow_grad_mag(flow_prev) + self._flow_grad_mag(flow_next))
        elif self.reliability_mode == "fb":
            b_prev = self.pair_flow(vi_curr, vi_prev)
            b_next = self.pair_flow(vi_curr, vi_next)
            err = 0.5 * ((flow_prev + flow_warp(b_prev, flow_prev)).abs().mean(dim=1, keepdim=True)
                         + (flow_next + flow_warp(b_next, flow_next)).abs().mean(dim=1, keepdim=True))
        else:
            return None
        rel = torch.exp(-self.rel_alpha * err)
        if self.rel_floor > 0:
            rel = rel.clamp(min=self.rel_floor)
        return rel

    def forward(self, sources_1, sources_2):
        b, t, c, h, w = sources_1.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            sources_1 = F.pad(sources_1, (0, pad_w, 0, pad_h))
            sources_2 = F.pad(sources_2, (0, pad_w, 0, pad_h))

        if self.input_restorer is not None:
            sources_1 = self.input_restorer(sources_1)
            sources_2 = self.input_restorer(sources_2)

        # Degradation descriptor: oracle (fed by the caller) or predicted from
        # the two raw modality stacks.
        dar_cond_ir = dar_cond_vi = None
        if self.dar_mode in ("oracle", "predicted"):
            if self.dar_mode == "oracle":
                dar_cond_ir = dar_cond_vi = self.dar_cond_vec
            elif self.dar_force_oracle and self.dar_cond_vec is not None:
                dar_cond_ir = dar_cond_vi = self.dar_cond_vec
            elif getattr(self.dar_estimator, "cross_modal", False):
                # one descriptor from both modalities: the IR/VIS relationship is
                # what disambiguates a darkened scene from a dark one
                dar_cond_ir = dar_cond_vi = self.dar_estimator(sources_1, sources_2)
            else:
                dar_cond_ir = self.dar_estimator(sources_1)
                dar_cond_vi = self.dar_estimator(sources_2)

        fused_results_list = []
        cached_flow_next = None
        # gate tensors are only produced when the gate is on; initialise them so a
        # normal (ungated) forward pass does not reference an unbound local
        dar_gate_ir = None
        dar_gate_vi = None

        for i in range(1, t - 1):
            # Sliding window: previous, current, next frames
            ir_prev, ir_curr, ir_next = sources_1[:, i - 1], sources_1[:, i], sources_1[:, i + 1]
            vi_prev, vi_curr, vi_next = sources_2[:, i - 1], sources_2[:, i], sources_2[:, i + 1]

            # Stage 1: shallow feature extraction
            f_ir_curr = self.feat_extractor_ir(ir_curr)
            f_ir_prev = self.feat_extractor_ir(ir_prev)
            f_ir_next = self.feat_extractor_ir(ir_next)

            f_vi_curr = self.feat_extractor_vi(vi_curr)
            f_vi_prev = self.feat_extractor_vi(vi_prev)
            f_vi_next = self.feat_extractor_vi(vi_next)

            # Stage 1.5: degradation-aware feature rectification (DAR).  Applied
            # to every frame of the window so the alignment stage sees rectified
            # features for prev/curr/next alike.
            if self.dar_ir is not None:
                # near-discrete condition: the form the oracle proved works.
                # Kept in loop-local variables so the descriptor is coded exactly
                # once per frame (coding it in place would compound across frames).
                if self.dar_vq > 0 and self.dar_estimator is not None:
                    c_ir_f = (self.dar_estimator.code(dar_cond_ir)
                              if dar_cond_ir is not None else None)
                    c_vi_f = (self.dar_estimator.code(dar_cond_vi)
                              if (dar_cond_vi is not None and dar_cond_vi is not dar_cond_ir)
                              else c_ir_f)
                else:
                    c_ir_f, c_vi_f = dar_cond_ir, dar_cond_vi
                if self.dar_gate and self.dar_estimator is not None:
                    # one gate value per modality, from the descriptor
                    g_ir = torch.sigmoid(self.dar_estimator.gate_head(dar_cond_ir))
                    g_vi = torch.sigmoid(self.dar_estimator.gate_head(dar_cond_vi))
                    self.last_gate = float(0.5 * (g_ir.mean() + g_vi.mean()))
                    dar_gate_ir, dar_gate_vi = g_ir, g_vi
                f_ir_curr = self.dar_ir(f_ir_curr, c_ir_f, dar_gate_ir)
                f_ir_prev = self.dar_ir(f_ir_prev, c_ir_f, dar_gate_ir)
                f_ir_next = self.dar_ir(f_ir_next, c_ir_f, dar_gate_ir)
                f_vi_curr = self.dar_vi(f_vi_curr, c_vi_f, dar_gate_vi)
                f_vi_prev = self.dar_vi(f_vi_prev, c_vi_f, dar_gate_vi)
                f_vi_next = self.dar_vi(f_vi_next, c_vi_f, dar_gate_vi)

            # Stage 2: optical flow + motion mask (cached across consecutive frames)
            if cached_flow_next is not None:
                flow_prev = cached_flow_next
            else:
                flow_prev = self.pair_flow(vi_prev, vi_curr)
            if self.flow_budget == "half" and cached_flow_next is not None:
                flow_next = cached_flow_next          # reuse instead of recomputing
            elif self.flow_budget == "single" and cached_flow_next is not None:
                flow_next = cached_flow_next
                flow_prev = cached_flow_next          # one call per window in total
            elif self.flow_budget == "single":
                flow_prev = self.pair_flow(vi_prev, vi_curr)
                flow_next = flow_prev
            else:
                flow_next = self.pair_flow(vi_next, vi_curr)
            cached_flow_next = flow_next

            motion_mask = self.generate_motion_mask_from_flow(flow_prev, flow_next)
            if self.reliability_mode != "none":
                rel = self.flow_reliability(flow_prev, flow_next, vi_prev, vi_curr, vi_next)
                if rel is not None:
                    motion_mask = motion_mask * rel

            # Stage 3: motion-aware alignment (cross-modal anchor)
            f_ir_final = self.aam_ir(
                f_curr=f_ir_curr, f_prev=f_ir_prev, f_next=f_ir_next,
                flow_prev=flow_prev, flow_next=flow_next,
                motion_mask=motion_mask, f_anchor=f_vi_curr
            )

            f_vi_final = self.aam_vi(
                f_curr=f_vi_curr, f_prev=f_vi_prev, f_next=f_vi_next,
                flow_prev=flow_prev, flow_next=flow_next,
                motion_mask=motion_mask, f_anchor=f_ir_curr
            )

            f_ir_final = f_ir_final + f_ir_curr
            f_vi_final = f_vi_final + f_vi_curr

            # Stage 3.5: second, conditioned rectification at the fusion-net input
            if self.dar_ir2 is not None:
                f_ir_final = self.dar_ir2(f_ir_final, dar_cond_ir)
                f_vi_final = self.dar_vi2(f_vi_final, dar_cond_vi)

            # Stage 4: dual-interaction fusion
            fused_frame = self.fusion_net(f_ir_final, f_vi_final, motion_mask)

            if pad_h or pad_w:
                fused_frame = fused_frame[..., :h, :w]

            fused_results_list.append(fused_frame)

        return torch.stack(fused_results_list, dim=1)
