%% HDO - 5 序列 MATLAB 评估（抽样 + 进度条）
clear; clc;
if isempty(gcp('nocreate')), parpool(64); end

root      = 'eval_data/HDO';
ir_dir    = fullfile(root, 'ir');
vis_dir   = fullfile(root, 'vis');
fused_dir = fullfile(root, 'fused');

target_seqs = {'seq_1', 'seq_2', 'seq_3', 'seq_4', 'seq_5'};
names = {'QG','QM','QP','QS','QCB','QABF','VIF'};
paper = [0.6629, 0.9966, 0.5660, 0.8464, 0.5578, 0.6837, 0.3359];

% 选取 5 个序列
all_files = dir(fullfile(fused_dir, '*.jpg'));
all_names = {all_files.name};
keep = false(size(all_names));
for s = 1:length(target_seqs)
    keep = keep | startsWith(all_names, [target_seqs{s} '_']);
end
files = all_files(keep(:));

% ============ 抽样：每 5 帧取 1 帧 ============
SAMPLE_STEP = 5;
files = files(1:SAMPLE_STEP:end);
n = length(files);
fprintf('HDO 5 序列，抽样后 %d 帧（原始 %d 帧）\n', n, length(all_files(keep(:))));

% ============ 进度条 ============
dq = parallel.pool.DataQueue;
afterEach(dq, @(~) fprintf('.'));
fprintf('开始计算，每帧输出一个点：\n');

% ============ 并行评估 ============
R = nan(n, length(names));
parfor i = 1:n
    fname = files(i).name;
    try
        I0 = im2double(rgb2gray(imread(fullfile(ir_dir,    fname))));
        V0 = im2double(rgb2gray(imread(fullfile(vis_dir,   fname))));
        F0 = im2double(rgb2gray(imread(fullfile(fused_dir, fname))));
        [H, W] = size(F0);
        if ~isequal(size(I0),[H,W]), I0 = imresize(I0,[H,W]); end
        if ~isequal(size(V0),[H,W]), V0 = imresize(V0,[H,W]); end
        I255 = I0*255; V255 = V0*255; F255 = F0*255;

        r = nan(1, length(names));
        try r(1) = metricXydeas(I0, V0, F0);                  catch, end
        try r(2) = metricPeilla(I0, V0, F0, 1);               catch, end
        try r(3) = metricZhao(I255, V255, F255);              catch, end
        try r(4) = metricPeilla(I255, V255, F255, 3);         catch, end
        try r(5) = metricChenBlum(I255, V255, F255);          catch, end
        try r(6) = metricXydeas(I255, V255, F255);            catch, end
        try r(7) = (vifvec(I255,F255)+vifvec(V255,F255))/2;   catch, end
        R(i,:) = r;
    catch
    end
    send(dq, i);   % 进度条
end
fprintf('\n');

% ============ 统计与输出 ============
m = nanmean(R, 1);
fprintf('\n=== HDO 5 序列平均 ===\n');
fprintf('%-6s %-10s %-10s %s\n', '指标','你的值','论文值','差异');
for k = 1:length(names)
    fprintf('%-6s %-10.4f %-10.4f %+.4f\n', ...
        names{k}, m(k), paper(k), m(k)-paper(k));
end

save('metrics_HDO_5seq.mat', 'R', 'names', 'm');
writetable(array2table([m; paper], 'VariableNames', names, ...
    'RowNames', {'Ours','Paper'}), 'metrics_HDO_5seq.csv', 'WriteRowNames', true);

fprintf('\n✅ 完成，结果保存至 metrics_HDO_5seq.csv\n');