clear; clc;
if isempty(gcp('nocreate')), parpool(64); end

datasets = {
    'HDO',   'eval_data/HDO';
    'M3SVD', 'eval_data/M3SVD';
    'VTMOT', 'eval_data/VTMOT';
};

names = {'QG','QM','QP','QS','QCB','QABF','VIF'};

for d = 1:size(datasets,1)
    ds_name = datasets{d,1};
    root    = datasets{d,2};
    ir_dir    = fullfile(root, 'ir');
    vis_dir   = fullfile(root, 'vis');
    fused_dir = fullfile(root, 'fused');

    files = dir(fullfile(fused_dir, '*.jpg'));
    if isempty(files), files = dir(fullfile(fused_dir, '*.png')); end
    files = files(1:3:end);
    n = length(files);
    fprintf('\n[%s] %d 帧\n', ds_name, n);

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

    fprintf('=== %s ===\n', ds_name);
    for k = 1:length(names)
        fprintf('  %-6s %.4f\n', names{k}, nanmean(R(:,k)));
    end
    save(sprintf('metrics_%s_matlab.mat', ds_name), 'R', 'names');
    writematrix(nanmean(R,1), sprintf('metrics_%s_matlab.csv', ds_name));
end