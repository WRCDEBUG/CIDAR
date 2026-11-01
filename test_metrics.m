%% 单张图测试各指标函数（修正尺寸）
clear; clc;

ir_path  = 'eval_data/HDO/ir/seq_1_000001.jpg';
vis_path = 'eval_data/HDO/vis/seq_1_000001.jpg';
fus_path = 'eval_data/HDO/fused/seq_1_000001.jpg';

I = double(rgb2gray(imread(ir_path)));
V = double(rgb2gray(imread(vis_path)));
F = double(rgb2gray(imread(fus_path)));

% 以融合图尺寸为准，对齐 IR 和 VIS
[H, W] = size(F);
if ~isequal(size(I), [H, W])
    I = imresize(I, [H, W]);
end
if ~isequal(size(V), [H, W])
    V = imresize(V, [H, W]);
end

fprintf('对齐后尺寸: IR=%s, VIS=%s, FUS=%s\n', ...
    mat2str(size(I)), mat2str(size(V)), mat2str(size(F)));

%% ---- 逐个测试 ----
test_list = {
    'metricXydeas',   'QAB/F';
    'metricHossny',   'QG';
    'metricPWW',      'QM';
    'metricZhao',     'QP';
    'metricPeilla',   'QS';
    'metricChenBlum', 'QCB';
    'metricMI',       'MI';
    'metricYang',     'QY';
    'metricZheng',    'QZheng';
    'Qabf',           'Qabf';
    'VIFF_Public',    'VIF';
};

for k = 1:size(test_list, 1)
    fname = test_list{k, 1};
    desc  = test_list{k, 2};
    fprintf('[%d/%d] %s (%s)...\n', k, size(test_list,1), fname, desc);
    try
        fh = str2func(fname);
        r = fh(I, V, F);
        if isnumeric(r)
            fprintf('     ✅ 结果: %.4f\n', r);
        elseif isstruct(r)
            disp(r);
        else
            disp(r);
        end
    catch ME
        fprintf('     ❌ 错误: %s\n', ME.message);
    end
end