%% 完整跑 seq_1 一个序列，对比论文参考值
clear; clc;

ir_dir    = 'eval_data/HDO/ir';
vis_dir   = 'eval_data/HDO/vis';
fused_dir = 'eval_data/HDO/fused';

files = dir(fullfile(fused_dir, 'seq_1_*.jpg'));
n = length(files);
fprintf('seq_1 共 %d 帧\n', n);

metric_names = {'metricXydeas', 'Qabf', 'metricPeilla1', 'metricPeilla3', ...
                'metricZhao', 'metricChenBlum', 'metricZheng', 'metricYang', ...
                'vifvec', 'VIFF_Public', 'metricPWW', 'metricWang'};
results = nan(n, length(metric_names));

for i = 1:n
    fname = files(i).name;
    I = double(rgb2gray(imread(fullfile(ir_dir,    fname))));
    V = double(rgb2gray(imread(fullfile(vis_dir,   fname))));
    F = double(rgb2gray(imread(fullfile(fused_dir, fname))));
    [H, W] = size(F);
    if ~isequal(size(I), [H, W]), I = imresize(I, [H, W]); end
    if ~isequal(size(V), [H, W]), V = imresize(V, [H, W]); end

    try results(i,1)  = metricXydeas(I, V, F);            catch, end
    try results(i,2)  = Qabf(I, V, F);                    catch, end
    try results(i,3)  = metricPeilla(I, V, F, 1);         catch, end
    try results(i,4)  = metricPeilla(I, V, F, 3);         catch, end
    try results(i,5)  = metricZhao(I, V, F);              catch, end
    try results(i,6)  = metricChenBlum(I, V, F);          catch, end
    try results(i,7)  = metricZheng(I, V, F);             catch, end
    try results(i,8)  = metricYang(I, V, F);              catch, end
    try results(i,9)  = vifvec(im2double(I), im2double(F)); catch, end
    try results(i,10) = VIFF_Public(I, V, F);             catch, end
    try results(i,11) = metricPWW(I, V, F);               catch, end
    try results(i,12) = metricWang(I, V, F);              catch, end

    if mod(i, 25) == 0, fprintf('  %d / %d\n', i, n); end
end

fprintf('\n=== seq_1 (%d 帧) 平均值 ===\n', n);
for k = 1:length(metric_names)
    fprintf('  %-20s  %.4f\n', metric_names{k}, nanmean(results(:,k)));
end

fprintf('\n=== 论文 HDO 参考值 ===\n');
fprintf('  QG    = 0.6629\n');
fprintf('  QM    = 0.9966\n');
fprintf('  QP    = 0.5660\n');
fprintf('  QS    = 0.8464\n');
fprintf('  QCB   = 0.5578\n');
fprintf('  VIF   = 0.3359\n');
fprintf('  QAB/F = 0.6837\n');
fprintf('  MS2R  = 0.9826\n');