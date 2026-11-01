%% ============================================================
%  MAVFusion 评估入口（MATLAB 侧）
%
%  本脚本负责 6 个图像指标：QG, QM, QP, QS, QCB, QABF
%  VIF 与 MS2R 由 Python 用 UniVF 参考实现计算：tools/run_univf_metrics.py
%   （UniVF = VF-Bench, https://github.com/Zhaozixiang1228/VF-Bench）
%  最后用 tools/merge_metrics.py 合并成 8 指标总表。
%
%  序列清单、评价路径与论文数值的唯一来源：
%      runs/00_contract/eval_manifest.json
%  度量映射的唯一来源：runs/00_contract/evaluation_contract.json
%
%  ⚠ 与原版 run_all_matlab.m 的三处差异（原版会给出错误数值）：
%     1) QM 原为 metricPeilla(I,V,F,1) —— 那实际是 QS。Q_M 应使用 metricPWW，
%        见 Objective-evaluation-for-image-fusion/fusionAssess.m:42-43 与
%        ../指标参数计算方法/evaluate_mavfusion_metrics.m:100。
%     2) QS 原为 metricPeilla(I255,V255,F255,3)；权威映射为 metricPeilla(...,1)。
%     3) QABF 原为 metricXydeas(I255,...) —— 那是 QG 的函数；Q^AB/F 应使用 Qabf()。
%     修正后 QABF 从 0.5567 变为 0.7396（论文 0.7550）。
%     4) QS 的输入尺度：metricPeilla 内部用 ssim_index，其稳定常数按 8-bit 设定。
%        传 [0,1] 会让 SSIM 饱和在 ~0.995，传 [0,255] 才得到论文量级。
%        探针（tools/metric_convention_probe.m）在三个数据集上的跨数据集最大偏差：
%        sw=1 [0,1] 0.1502 | sw=1 [0,255] 0.0170 | sw=3 [0,255] 0.0353
%        因此采用 sw=1 + [0,255]。
%
%  用法： matlab -batch "run('VIF等参数计算/run_all_matlab.m')"
% ============================================================
clear; clc;

REPO = fileparts(fileparts(mfilename('fullpath')));   % 本文件在 VIF等参数计算/ 下
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion'));
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion', 'VIF', 'matlabPyrTools-master'));
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion', 'VIF', 'VIF'));

manifest = jsondecode(fileread(fullfile(REPO, 'runs', '00_contract', 'eval_manifest.json')));

SAMPLE_STEP = 5;      % 与历史 run_*.m 的抽样步长一致；配对比较必须对所有方法一致
OUT_DIR = fullfile(REPO, 'output', 'pr1', 'matlab');
if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end

if isempty(gcp('nocreate'))
    parpool(min(32, feature('numcores')));
end

names = {'QG','QM','QP','QS','QCB','QABF'};   % VIF 与 MS2R 由 Python(UniVF) 提供
ds_list = fieldnames(manifest.datasets);

fprintf('\n================ MAVFusion 图像指标（MATLAB） ================\n');
fprintf('抽样步长 %d，视频等权；映射见 evaluation_contract.json\n', SAMPLE_STEP);

all_rows = {};

for d = 1:numel(ds_list)
    ds_name = ds_list{d};
    ds = manifest.datasets.(ds_name);
    set_name = ds.default_set;
    seqs = manifest.sequence_sets.(set_name).sequences;
    if ischar(seqs), seqs = cellstr(seqs); end
    if isstring(seqs), seqs = cellstr(seqs); end
    if iscell(seqs) && numel(seqs) == 1 && iscell(seqs{1}), seqs = seqs{1}; end

    root = fullfile(REPO, ds.eval_root);
    ir_dir = fullfile(root, 'ir'); vis_dir = fullfile(root, 'vis'); fused_dir = fullfile(root, 'fused');
    if ~isfolder(fused_dir)
        fprintf('[%s] 缺少 %s，跳过\n', ds_name, fused_dir); continue;
    end

    % 扩展名自动探测（原版把 M3SVD 写死为 .png，实际文件是 .jpg）
    if ~isempty(dir(fullfile(fused_dir, '*.jpg'))), ext = '.jpg';
    elseif ~isempty(dir(fullfile(fused_dir, '*.png'))), ext = '.png';
    else, fprintf('[%s] fused 目录无图，跳过\n', ds_name); continue; end

    % 按序列收集帧
    seq_of = {}; frame_of = {}; files = {};
    for s = 1:numel(seqs)
        f = dir(fullfile(fused_dir, [seqs{s} '_*' ext]));
        if isempty(f)
            fprintf('  [warn] %s: 未找到 %s 的帧\n', ds_name, seqs{s}); continue;
        end
        [~, order] = sort({f.name}); f = f(order);
        idx = 1:SAMPLE_STEP:numel(f);
        for j = idx
            files{end+1} = f(j).name;      %#ok<SAGROW>
            seq_of{end+1} = seqs{s};       %#ok<SAGROW>
        end
    end
    n = numel(files);
    fprintf('\n[%s] 序列 %d 条，抽样后 %d 帧（步长 %d）\n', ds_name, numel(seqs), n, SAMPLE_STEP);
    if n == 0, continue; end

    R = nan(n, numel(names));
    parfor i = 1:n
        fn = files{i};
        r = nan(1, numel(names));
        try
            I0 = im2double(rgb2gray(imread(fullfile(ir_dir,    fn))));
            V0 = im2double(rgb2gray(imread(fullfile(vis_dir,   fn))));
            F0 = im2double(rgb2gray(imread(fullfile(fused_dir, fn))));
            [H, W] = size(F0);
            if ~isequal(size(I0),[H,W]), I0 = imresize(I0,[H,W]); end
            if ~isequal(size(V0),[H,W]), V0 = imresize(V0,[H,W]); end
            I255 = I0*255; V255 = V0*255; F255 = F0*255;
            r(1) = metricXydeas(I0, V0, F0);                    % QG   [0,1]
            r(2) = metricPWW(I255, V255, F255);                 % QM   [0,255]  ← 修正
            r(3) = metricZhao(I255, V255, F255);                % QP   [0,255]
            r(4) = metricPeilla(I255, V255, F255, 1);            % QS   [0,255], sw=1 ← 修正
            r(5) = metricChenBlum(I255, V255, F255);            % QCB  [0,255]
            r(6) = Qabf(I255, V255, F255);                      % QABF [0,255] ← 修正
            % VIF 不在此处计算：UniVF 参考实现是 Python 的 Metric_VIF（vifp），
            % MATLAB 的 vifvec 是另一套实现，与论文一致偏高 +0.10~0.20。
            % 见 tools/run_univf_metrics.py
        catch ME
            fprintf('  [error] %s: %s\n', fn, ME.message);
        end
        R(i,:) = r;
    end

    % ---- 逐序列均值 → 视频等权 ----
    seq_names = unique(seq_of, 'stable');
    per_seq = nan(numel(seq_names), numel(names));
    for k = 1:numel(seq_names)
        per_seq(k,:) = nanmean(R(strcmp(seq_of, seq_names{k}), :), 1);
    end
    video_equal = nanmean(per_seq, 1);
    frame_equal = nanmean(R, 1);
    paper_vec = zeros(1, numel(names));
    for k = 1:numel(names), paper_vec(k) = ds.paper_table1.(names{k}); end

    fprintf('%-8s %-10s %-10s %s\n', '指标', '本地(视频等权)', '论文', '差异');
    for k = 1:numel(names)
        fprintf('%-8s %-14.4f %-10.4f %+.4f\n', names{k}, video_equal(k), paper_vec(k), video_equal(k)-paper_vec(k));
    end
    fprintf('  (帧等权 QABF=%.4f  QM=%.4f)\n', frame_equal(6), frame_equal(2));

    % ---- 逐帧 CSV（供合并与逐视频分析）----
    T = array2table(R, 'VariableNames', names);
    T = addvars(T, repmat({ds_name}, n, 1), seq_of(:), files(:), ...
        'Before', 1, 'NewVariableNames', {'dataset','sequence','frame'});
    writetable(T, fullfile(OUT_DIR, sprintf('per_frame_%s_%s.csv', ds_name, set_name)));

    % ---- 逐序列 CSV ----
    S = array2table(per_seq, 'VariableNames', names);
    S = addvars(S, repmat({ds_name}, numel(seq_names), 1), seq_names(:), ...
        'Before', 1, 'NewVariableNames', {'dataset','sequence'});
    writetable(S, fullfile(OUT_DIR, sprintf('per_sequence_%s_%s.csv', ds_name, set_name)));

    % ---- 汇总 CSV / MAT ----
    summary_tbl = array2table([video_equal; frame_equal; paper_vec], ...
        'VariableNames', names, 'RowNames', {'video_equal','frame_equal','paper'});
    writetable(summary_tbl, fullfile(OUT_DIR, sprintf('metrics_%s_%s.csv', ds_name, set_name)), ...
        'WriteRowNames', true);
    save(fullfile(OUT_DIR, sprintf('metrics_%s_%s.mat', ds_name, set_name)), ...
        'R', 'names', 'per_seq', 'seq_names', 'video_equal', 'frame_equal', 'set_name');

    all_rows{end+1} = struct('dataset', ds_name, 'set', set_name, ...
        'video_equal', video_equal, 'paper', paper_vec, 'frames', n); %#ok<SAGROW>
end

fprintf('\n================ 汇总（视频等权） ================\n');
fprintf('%-8s %-22s %8s %8s %8s %8s %8s %8s\n', ...
    'dataset','set','QG','QM','QP','QS','QCB','QABF');
for i = 1:numel(all_rows)
    a = all_rows{i};
    fprintf('%-8s %-22s', a.dataset, a.set);
    fprintf(' %8.4f', a.video_equal);
    fprintf('\n');
end
fprintf('\n产物目录：%s\n', OUT_DIR);
fprintf('VIF/MS2R 请运行：python tools/run_univf_metrics.py（UniVF 参考实现）\n');
fprintf('合并总表：python tools/merge_metrics.py\n');
fprintf('✅ MATLAB 部分完成\n');
