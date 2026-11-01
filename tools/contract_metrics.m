function contract_metrics(manifest_path, out_dir, sets_csv, sample_step, max_frames_per_seq)
% CONTRACT_METRICS  Frozen evaluation contract for MAVFusion (Step 0).
%
%   contract_metrics(manifest_path, out_dir, sets_csv, sample_step, max_per_seq)
%
%   Reads sequence sets and metric definitions from the manifest JSON, then
%   evaluates every fused frame with the pinned implementation and input scale
%   of each metric.  Outputs per-frame CSV plus per-sequence / video-equal
%   aggregate JSON.
%
%   The mapping in runs/00_contract/eval_manifest.json is the ONLY sanctioned
%   one.  It intentionally differs from VIF等参数计算/run_*.m, which filled the
%   QM column with metricPeilla(...,1) (i.e. QS) and the QABF column with
%   metricXydeas.  See runs/00_contract/evaluation_contract.json.

if nargin < 3 || isempty(sets_csv), sets_csv = 'frozen5_test'; end
if nargin < 4 || isempty(sample_step), sample_step = 5; end
if nargin < 5, max_frames_per_seq = inf; end

P = fileparts(fileparts(mfilename('fullpath')));   % repo root
OBJ = fullfile(P, 'Objective-evaluation-for-image-fusion');
addpath(OBJ);
addpath(fullfile(OBJ, 'VIF', 'matlabPyrTools-master'));
addpath(fullfile(OBJ, 'VIF', 'VIF'));

manifest = jsondecode(fileread(manifest_path));
% fullfile() concatenates rather than resolving an absolute second argument,
% so an absolute eval_root must be used verbatim.
if ~isempty(manifest.eval_root) && manifest.eval_root(1) == '/'
    root = manifest.eval_root;
else
    root = fullfile(P, manifest.eval_root);
end
if ~isfolder(out_dir), mkdir(out_dir); end

if isempty(gcp('nocreate'))
    parpool(min(32, feature('numcores')));
end

metric_names = {'QG','QABF','QM','QP','QS','QCB','VIF'};
sets = strtrim(strsplit(sets_csv, ','));

for si = 1:numel(sets)
    set_name = sets{si};
    if ~isfield(manifest.sequence_sets, set_name)
        error('Unknown sequence set: %s', set_name);
    end
    seqs = manifest.sequence_sets.(set_name).sequences;
    if ischar(seqs), seqs = cellstr(seqs); end
    if isstring(seqs), seqs = cellstr(seqs); end

    % ---- enumerate frames ------------------------------------------------
    seq_idx = []; names = {}; seq_of = {};
    for k = 1:numel(seqs)
        seq = seqs{k};
        d = dir(fullfile(root, 'fused', [seq '_*.jpg']));
        if isempty(d)
            fprintf('  [warn] no fused frames for %s\n', seq);
            continue;
        end
        [~, order] = sort({d.name});
        d = d(order);
        idx = 1:sample_step:numel(d);
        if isfinite(max_frames_per_seq), idx = idx(1:min(numel(idx), max_frames_per_seq)); end
        for j = idx
            seq_idx(end+1) = k;              %#ok<AGROW>
            names{end+1} = d(j).name;        %#ok<AGROW>
            seq_of{end+1} = seq;             %#ok<AGROW>
        end
    end
    n = numel(names);
    fprintf('[%s] %d sequences, %d frames sampled (step=%d, cap=%g)\n', ...
        set_name, numel(seqs), n, sample_step, max_frames_per_seq);

    R = nan(n, numel(metric_names));
    elapsed = nan(n, 1);

    parfor i = 1:n
        fn = names{i};
        t0 = tic;
        r = nan(1, numel(metric_names));
        try
            I = im2double(rgb2gray(imread(fullfile(root, 'ir', fn))));
            V = im2double(rgb2gray(imread(fullfile(root, 'vis', fn))));
            F = im2double(rgb2gray(imread(fullfile(root, 'fused', fn))));
            [H, W] = size(F);
            if ~isequal(size(I), [H W]), I = imresize(I, [H W]); end
            if ~isequal(size(V), [H W]), V = imresize(V, [H W]); end
            I255 = I * 255; V255 = V * 255; F255 = F * 255;

            r(1) = metricXydeas(I, V, F);                    % QG   [0,1]
            r(2) = Qabf(I255, V255, F255);                   % QABF [0,255]
            r(3) = metricPWW(I255, V255, F255);              % QM   [0,255]
            r(4) = metricZhao(I255, V255, F255);             % QP   [0,255]
            r(5) = metricPeilla(I255, V255, F255, 1);        % QS   [0,255], sw=1 (manifest QS convention)
            r(6) = metricChenBlum(I255, V255, F255);         % QCB  [0,255]
            r(7) = (vifvec(I255, F255) + vifvec(V255, F255)) / 2;   % VIF [0,255]
        catch ME
            fprintf('  [error] %s: %s\n', fn, ME.message);
        end
        R(i, :) = r;
        elapsed(i) = toc(t0);
    end

    % ---- per-sequence, then video-equal --------------------------------
    per_seq = struct();
    seq_names = unique(seq_of, 'stable');
    video_equal = nan(1, numel(metric_names));
    for k = 1:numel(seq_names)
        mask = strcmp(seq_of, seq_names{k});
        vals = R(mask, :);
        per_seq.(matlab.lang.makeValidName(seq_names{k})) = nanmean(vals, 1);
    end
    for m = 1:numel(metric_names)
        col = nan(numel(seq_names), 1);
        for k = 1:numel(seq_names)
            col(k) = per_seq.(matlab.lang.makeValidName(seq_names{k}))(m);
        end
        video_equal(m) = nanmean(col);
    end

    payload = struct();
    payload.set = set_name;
    payload.sequences = seq_names;
    payload.per_sequence = per_seq;
    payload.video_equal_mean = array2struct(metric_names, video_equal);
    payload.frame_equal_mean = array2struct(metric_names, nanmean(R, 1));
    payload.frames = n;
    payload.sample_step = sample_step;
    payload.seconds_total = nansum(elapsed);
    payload.seconds_median_per_frame = nanmedian(elapsed);
    payload.manifest = manifest_path;
    payload.metric_contract = manifest.metrics;

    json_path = fullfile(out_dir, sprintf('contract_metrics_%s.json', set_name));
    fid = fopen(json_path, 'w');
    fprintf(fid, '%s', jsonencode(payload, 'PrettyPrint', true));
    fclose(fid);

    csv_path = fullfile(out_dir, sprintf('contract_metrics_%s_per_frame.csv', set_name));
    T = array2table(R, 'VariableNames', metric_names);
    T = addvars(T, seq_of(:), names(:), 'Before', 1, 'NewVariableNames', {'sequence', 'frame'});
    writetable(T, csv_path);

    fprintf('[%s] video-equal:', set_name);
    for m = 1:numel(metric_names)
        fprintf('  %s=%.4f', metric_names{m}, video_equal(m));
    end
    fprintf('\n  per-sequence means:\n');
    for k = 1:numel(seq_names)
        v = per_seq.(matlab.lang.makeValidName(seq_names{k}));
        fprintf('    %-12s', seq_names{k});
        for m = 1:numel(metric_names), fprintf(' %.4f', v(m)); end
        fprintf('\n');
    end
    fprintf('  -> %s\n  -> %s\n', json_path, csv_path);
end

fprintf('\nDONE contract_metrics\n');
end


function s = array2struct(names, values)
s = struct();
for i = 1:numel(names)
    s.(names{i}) = values(i);
end
end
