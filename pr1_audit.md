# PR1 audit

## Frozen implementation notes

- The local implementation is the reference for this audit. The existing
  `src/model/fusion.py` uses `adaptive_avg_pool2d(...).view(-1)`,
  `k = max(1, int(0.3 * scores.numel()))`, unsorted `torch.topk`,
  `soft_weights = val`, and zero-scatter before `HybridBlock` applies
  `d_feat * m_s` and `smooth`.
- `MaskedDATBlock` keeps its attention, `topk=64`, zero-fill before softmax,
  and all checkpoint weights are untouched.
- PR1 adds an optional selector only at the requested `Fusion_Net` block. With
  the default `selector=None`, the original Top-K path is unchanged.
- `Q_M` is not implemented in `src/util/metric.py`. The repository's existing
  MATLAB evaluator identifies `Objective-evaluation-for-image-fusion/metricPWW.m`
  as the PWW multi-scale metric and `fusionAssess.m` labels it `Q_M`; PR1 uses
  that implementation rather than rewriting the metric.

## Known repository difference

- The prepared `data_split/IVF/M3SVD/split.json` contains all 30 sequences in
  `test` and no `train` sequences. The five sequences already used by the local
  M3SVD reproduction are frozen as the test holdout; the remaining 25 are the
  PR1 development videos.
- The local Python evaluator reports VIF/SSIM/MI/Qabf and does not provide
  `Q_M`; therefore no Python proxy is used for the PR1 primary decision.

## Gate condition

If the legacy equivalence check is not below `1e-5`, statistics and tie-break
experiments must not be treated as valid.
