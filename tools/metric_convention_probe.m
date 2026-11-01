%% 度量约定探针：为 QS 与 VIF 找出与论文最一致的实现约定
%
% 背景：按 evaluation_contract.json 的映射，QG/QM/QP/QCB/QABF 与论文的差异已在
% 0.002–0.05 内，但 QS 在三个数据集上一致偏高 +0.14、VIF 一致偏高 +0.10~0.20。
% 这种跨数据集一致的偏移说明是约定问题（输入尺度或变体选择），不是序列噪声。
%
% 本脚本在少量帧上枚举候选约定，与论文表 1 对比，输出各候选的跨数据集一致性。
%
% 用法： matlab -batch "run('tools/metric_convention_probe.m')"

clear; clc;
REPO = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion'));
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion', 'VIF', 'matlabPyrTools-master'));
addpath(fullfile(REPO, 'Objective-evaluation-for-image-fusion', 'VIF', 'VIF'));

manifest = jsondecode(fileread(fullfile(REPO, 'runs', '00_contract', 'eval_manifest.json')));
ds_list = fieldnames(manifest.datasets);
MAX_FRAMES = 12;

qs_names  = {'peilla_sw1_01','peilla_sw1_255','peilla_sw3_01','peilla_sw3_255'};
vif_names = {'vif_both_255','vif_ir_255','vif_both_01','vif_ir_01'};
paper_qs  = struct(); paper_vif = struct();

QS  = nan(numel(ds_list), numel(qs_names));
VIF = nan(numel(ds_list), numel(vif_names));

for d = 1:numel(ds_list)
    ds_name = ds_list{d};
    ds = manifest.datasets.(ds_name);
    seqs = manifest.sequence_sets.(ds.default_set).sequences;
    if ischar(seqs), seqs = cellstr(seqs); end
    root = fullfile(REPO, ds.eval_root);
    if ~isempty(dir(fullfile(root,'fused','*.jpg'))), ext='.jpg'; else, ext='.png'; end
    files = {};
    for s = 1:numel(seqs)
        f = dir(fullfile(root,'fused',[seqs{s} '_*' ext]));
        if isempty(f), continue; end
        f = f(1:min(3,numel(f)));
        for j = 1:numel(f), files{end+1} = f(j).name; end %#ok<SAGROW>
    end
    files = files(1:min(MAX_FRAMES, numel(files)));
    n = numel(files);
    fprintf('[%s] 探针帧 %d\n', ds_name, n);

    Rq = nan(n, numel(qs_names));
    Rv = nan(n, numel(vif_names));
    for i = 1:n
        fn = files{i};
        I0 = im2double(rgb2gray(imread(fullfile(root,'ir',fn))));
        V0 = im2double(rgb2gray(imread(fullfile(root,'vis',fn))));
        F0 = im2double(rgb2gray(imread(fullfile(root,'fused',fn))));
        [H,W] = size(F0);
        if ~isequal(size(I0),[H W]), I0 = imresize(I0,[H W]); end
        if ~isequal(size(V0),[H W]), V0 = imresize(V0,[H W]); end
        I255 = I0*255; V255 = V0*255; F255 = F0*255;
        try, Rq(i,1) = metricPeilla(I0,  V0,  F0,  1); catch, end
        try, Rq(i,2) = metricPeilla(I255,V255,F255,1); catch, end
        try, Rq(i,3) = metricPeilla(I0,  V0,  F0,  3); catch, end
        try, Rq(i,4) = metricPeilla(I255,V255,F255,3); catch, end
        try, Rv(i,1) = (vifvec(I255,F255)+vifvec(V255,F255))/2; catch, end
        try, Rv(i,2) = vifvec(I255,F255); catch, end
        try, Rv(i,3) = (vifvec(I0,F0)+vifvec(V0,F0))/2; catch, end
        try, Rv(i,4) = vifvec(I0,F0); catch, end
    end
    QS(d,:) = nanmean(Rq,1);
    VIF(d,:) = nanmean(Rv,1);
    paper_qs.(ds_name) = ds.paper_table1.QS;
    paper_vif.(ds_name) = ds.paper_table1.VIF;
end

fprintf('\n================ QS 候选（论文值 vs 各约定）================\n');
fprintf('%-8s %8s', 'dataset', 'paper');
for k=1:numel(qs_names), fprintf(' %16s', qs_names{k}); end
fprintf('\n');
for d=1:numel(ds_list)
    ds_name = ds_list{d};
    fprintf('%-8s %8.4f', ds_name, paper_qs.(ds_name));
    for k=1:numel(qs_names), fprintf(' %16.4f', QS(d,k)); end
    fprintf('\n');
end
fprintf('%-8s %8s', 'max|diff|', '-');
for k=1:numel(qs_names)
    fprintf(' %16.4f', max(abs(QS(:,k) - cellfun(@(x) paper_qs.(x), ds_list))));
end
fprintf('   <-- 越小越一致\n');

fprintf('\n================ VIF 候选 ================\n');
fprintf('%-8s %8s', 'dataset', 'paper');
for k=1:numel(vif_names), fprintf(' %14s', vif_names{k}); end
fprintf('\n');
for d=1:numel(ds_list)
    ds_name = ds_list{d};
    fprintf('%-8s %8.4f', ds_name, paper_vif.(ds_name));
    for k=1:numel(vif_names), fprintf(' %14.4f', VIF(d,k)); end
    fprintf('\n');
end
fprintf('%-8s %8s', 'max|diff|', '-');
for k=1:numel(vif_names)
    fprintf(' %14.4f', max(abs(VIF(:,k) - cellfun(@(x) paper_vif.(x), ds_list))));
end
fprintf('   <-- 越小越一致\n');

fprintf('\n说明：跨数据集一致的偏移说明是约定问题。若某候选在三个数据集上同时最接近论文，\n');
fprintf('应把它写进 runs/00_contract/evaluation_contract.json 的 metrics 段并重跑 run_all_matlab.m。\n');
