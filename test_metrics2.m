%% 扩展测试：确认 QS/QG/QM/VIF 的对应关系
clear; clc;

ir_path  = 'eval_data/HDO/ir/seq_1_000001.jpg';
vis_path = 'eval_data/HDO/vis/seq_1_000001.jpg';
fus_path = 'eval_data/HDO/fused/seq_1_000001.jpg';

I = double(rgb2gray(imread(ir_path)));
V = double(rgb2gray(imread(vis_path)));
F = double(rgb2gray(imread(fus_path)));

[H, W] = size(F);
I = imresize(I, [H, W]);
V = imresize(V, [H, W]);

fprintf('=== QS: metricPeilla 三种 sw ===\n');
for sw = 1:3
    try
        r = metricPeilla(I, V, F, sw);
        fprintf('  metricPeilla(sw=%d) = %.4f\n', sw, r);
    catch ME
        fprintf('  metricPeilla(sw=%d) 错误: %s\n', sw, ME.message);
    end
end

fprintf('\n=== QG: 搜 QG 相关函数 ===\n');
% 看 evalution.m 里 QG 怎么算的
!head -80 Objective-evaluation-for-image-fusion/evalution.m

fprintf('\n=== VIF: 试几个可能的函数 ===\n');
vif_candidates = {'vifp_mscale', 'vifvec', 'vifp', 'VIF', 'metricVIF'};
for k = 1:length(vif_candidates)
    fname = vif_candidates{k};
    try
        fh = str2func(fname);
        r = fh(I, F);  % VIF 通常只需 2 个参数
        fprintf('  ✅ %s = %.4f\n', fname, r);
    catch
        fprintf('  ❌ %s 不可用\n', fname);
    end
end

fprintf('\n=== QG 候选: 试几个可能的函数 ===\n');
qg_candidates = {'metricQG', 'QG', 'metricHossny', 'metricWang'};
for k = 1:length(qg_candidates)
    fname = qg_candidates{k};
    try
        fh = str2func(fname);
        r = fh(I, V, F);
        fprintf('  ✅ %s = %.4f\n', fname, r);
    catch
        fprintf('  ❌ %s 不可用\n', fname);
    end
end