clear; clc;
if isempty(gcp('nocreate')), parpool(64); end

root      = 'eval_data/HDO';
ir_dir    = fullfile(root, 'ir');
vis_dir   = fullfile(root, 'vis');
fused_dir = fullfile(root, 'fused');

% 只选 seq_1 ~ seq_5
target_seqs = {'seq_1_', 'seq_2_', 'seq_3_', 'seq_4_', 'seq_5_'};
all_files = dir(fullfile(fused_dir, '*.jpg'));
all_names = {all_files.name};

% 关键修复：确保向量方向一致
keep = false(size(all_names));       % 与 all_names 同形状
for s = 1:length(target_seqs)
    keep = keep | startsWith(all_names, target_seqs{s});
end
files = all_files(keep(:));          % 强制列向量索引

files = files(1:3:end);
n = length(files);
fprintf('HDO 5 序列，抽样后 %d 帧\n', n);

names = {'QG','QM','QP','QS','QCB','QABF','VIF'};
R = nan(n, length(names));

parfor i = 1:n
    fname = files(i).name;
    I0 = im2double(rgb2gray(imread(fullfile(ir_dir,    fname))));
    V0 = im2double(rgb2gray(imread(fullfile(vis_dir,   fname))));
    F0 = im2double(rgb2gray(imread(fullfile(fused_dir, fname))));
    [H, W] = size(F0);
    I0 = imresize(I0,[H,W]); V0 = imresize(V0,[H,W]);
    I255 = I0*255; V255 = V0*255; F255 = F0*255;

    r = nan(1, length(names));
    try r(1) = metricXydeas(I0, V0, F0);           catch, end
    try r(2) = metricPeilla(I0, V0, F0, 1);        catch, end
    try r(3) = metricZhao(I255, V255, F255);       catch, end
    try r(4) = metricPeilla(I255, V255, F255, 3);  catch, end
    try r(5) = metricChenBlum(I255, V255, F255);   catch, end
    try r(6) = metricXydeas(I255, V255, F255);     catch, end
    try r(7) = (vifvec(I255,F255)+vifvec(V255,F255))/2; catch, end
    R(i,:) = r;
end

paper = [0.6629, 0.9966, 0.5660, 0.8464, 0.5578, 0.6837, 0.3359];
fprintf('\n=== HDO 5 序列平均 ===\n');
fprintf('%-6s %-10s %-10s %s\n', '指标','你的值','论文值','差异');
for k = 1:length(names)
    m = nanmean(R(:,k));
    fprintf('%-6s %-10.4f %-10.4f %+.4f\n', names{k}, m, paper(k), m-paper(k));
end

save('metrics_HDO_5seq.mat', 'R', 'names');
writematrix([nanmean(R,1); paper], 'metrics_HDO_5seq.csv');