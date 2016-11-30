function [save_model_path, perf, cache_dir, db_train_path, db_val_path] = ...
    fast_rcnn_train(conf, imdb_train, roidb_train, varargin)
% save_model_path = fast_rcnn_train(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                               @isstruct);
    ip.addRequired('imdb_train',                         @iscell);
    ip.addRequired('roidb_train',                        @iscell);
    ip.addParameter('do_val',            false,          @isscalar);
    ip.addParameter('imdb_val',                          @iscell);
    ip.addParameter('roidb_val',                         @iscell);
    ip.addParameter('val_iters',         500,            @isscalar); 
    ip.addParameter('val_interval',      2000,           @isscalar); 
    ip.addParameter('snapshot_interval', 10000,          @isscalar);
    ip.addParameter('solver_def_file',   fullfile(pwd, 'models', 'Zeiler_conv5', 'solver.prototxt'),    @isstr);
    ip.addParameter('net_def_file',      fullfile(pwd, 'models', 'Zeiler_conv5', 'train_val.prototxt'), @isstr);                                                    
    ip.addParameter('net_file',          fullfile(pwd, 'models', 'Zeiler_conv5', 'Zeiler_conv5'),       @isstr);
    ip.addParameter('cache_name',        'Zeiler_conv5', @isstr);
    ip.addParameter('shouldContinue',    false,          @isscalar);
    
    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    
%% try to find trained model
    perf = {};
    
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    imdbs_name_val = cell2mat(cellfun(@(x) x.name, opts.imdb_val, 'UniformOutput', false));
    
    cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdbs_name);
    cache_dir_imdb = fullfile(pwd, 'imdb', 'cache');
    db_train_path = fullfile(cache_dir_imdb, ['imroidb_' imdbs_name '.mat']);
    db_val_path = fullfile(cache_dir_imdb, ['imroidb_' imdbs_name_val '.mat']);
    
    save_model_path = fullfile(cache_dir, 'final');
    perf_path = fullfile(cache_dir, 'perf.mat');
    if exist(save_model_path, 'file') && ~opts.shouldContinue
        if exist(perf_path, 'file')
            load(perf_path, 'perf');
        end
        return;
    end
    
%% init
    % init caffe solver
    mkdir_if_missing(cache_dir);
    
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    
    % we don't care about 'net' field in solver file
    % caffe_solver.net = caffe.Net(opts.net_def_file, 'train');
    
    if opts.shouldContinue && exist(save_model_path, 'file')
        % initalize from last saved model       
        saved_models = dir([save_model_path '*']);
        stage_num = length(saved_models) + 1;

        % model files
        saved_models_names = dir(fullfile(cache_dir, 'iter_*'));
        saved_models_names = {saved_models_names.name};
        saved_models_names{end+1} = 'final';
        for i_file = 1:length(saved_models_names)
            modified_name = sprintf('%02d_%s', ...
                stage_num - 1, saved_models_names{i_file});
            movefile(fullfile(cache_dir, saved_models_names{i_file}), ...
                fullfile(cache_dir, modified_name));
        end
        opts.net_file = fullfile(cache_dir, modified_name);

        % solver file
        if exist(fullfile(cache_dir, 'solver.prototxt'), 'file')
            movefile(fullfile(cache_dir, 'solver.prototxt'), ...
                fullfile(cache_dir, sprintf('%02d_solver.prototxt', stage_num - 1)));
        end

        % perf file
        if exist(perf_path, 'file')
            movefile(perf_path, ...
                strrep(perf_path, 'perf.mat', sprintf('%02d_perf.mat', stage_num - 1)));
        end
    end
    
    if ~strcmp(opts.net_file, '')
        % initialize from pre-trained network
        caffe_solver.net.copy_from(opts.net_file);
    end

    % first copy solver & network file to output dir
    copyfile(opts.net_def_file, fullfile(cache_dir, 'train_val.prototxt'));
    copyfile(opts.solver_def_file, fullfile(cache_dir, 'solver.prototxt'));
    mean_image = conf.image_means;
    save(fullfile(cache_dir, 'mean_value.mat'), 'mean_image');
    
    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);
    
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    
%% making tran/val data
    fprintf('Preparing training data...');
    if exist(db_train_path, 'file')
        load(db_train_path, 'image_roidb_train', 'bbox_means', 'bbox_stds');
    else
        [image_roidb_train, bbox_means, bbox_stds]...
            = fast_rcnn_prepare_image_roidb(conf, imdb_train, roidb_train);
        save(db_train_path, 'image_roidb_train', 'bbox_means', 'bbox_stds');
    end
    fprintf('Done.\n');
    
    if opts.do_val
        fprintf('Preparing validation data...');
        if exist(db_val_path, 'file')
            load(db_val_path, 'image_roidb_val');
        else
            [image_roidb_val]...
                = fast_rcnn_prepare_image_roidb(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
            save(db_val_path, 'image_roidb_val');
        end
        fprintf('Done.\n');

        % fix validation data
        shuffled_inds_val = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch, conf.batch_size);
        shuffled_inds_val = shuffled_inds_val(randperm(length(shuffled_inds_val), min(opts.val_iters, length(shuffled_inds_val))));
    end
%% update scale values
    if conf.keep_scale
        % set scales based on images dimensions
        im_size_all = reshape([image_roidb_train.im_size]', [2 length(image_roidb_train)]);
        conf.scales = unique(min(im_size_all, [], 1));
    end

%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  
    num_classes = size(image_roidb_train(1).overlap, 2);
    check_gpu_memory(conf, caffe_solver, num_classes, opts.do_val);
    
%% training
    shuffled_inds = [];
    train_results = [];  
    val_results = [];  
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();
    while (iter_ < max_iter)
        caffe_solver.net.set_phase('train');

        % generate minibatch training data
        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ...
            conf.ims_per_batch, conf.batch_size);
        [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
            fast_rcnn_get_minibatch(conf, image_roidb_train(sub_db_inds));
    
        net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
        if (length(caffe_solver.net.inputs) == 6)
            % should supply bbox_loss_weights_outside also (newer code of
            % smooth_L1_loss implementation)
            % we just set 1 for all out weights to get the same behaviour
            % as the older (original) code
            net_inputs{end+1} = ones(size(bbox_loss_weights_blob));
        end
        
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        
        rst = caffe_solver.net.get_output();
        train_results = parse_rst(train_results, rst);
            
        % do valdiation per val_interval iterations or 
        %                   once full epoch is done
        if ~mod(iter_, opts.val_interval) || isempty(shuffled_inds)
            if opts.do_val
                % PAY ATTENTION : this call doesn't really switch to full
                % test mode. Some layers, such as batch_norm, define
                % internal parameters according to train/test mode during
                % setup only (such as use_global_stats parameter)
                caffe_solver.net.set_phase('test');                
                for i = 1:size(shuffled_inds_val, 2)
                    sub_db_inds = shuffled_inds_val{i};
                    [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
                        fast_rcnn_get_minibatch(conf, image_roidb_val(sub_db_inds));

                    % Reshape net's input blobs
                    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
                    if (length(caffe_solver.net.inputs) == 6)
                        net_inputs{end+1} = ones(size(bbox_loss_weights_blob));
                    end
        
                    caffe_solver.net.reshape_as_input(net_inputs);
                    
                    caffe_solver.net.forward(net_inputs);
                    
                    rst = caffe_solver.net.get_output();
                    val_results = parse_rst(val_results, rst);
                end
            end
            
            perf_temp = show_state(iter_, train_results, val_results);
            perf_temp.iter = iter_;
            perf{end+1} = perf_temp;
            save(perf_path, 'perf');
            
            train_results = [];
            val_results = [];
            diary; diary; % flush diary
        end
        
        % snapshot
        if ~mod(iter_, opts.snapshot_interval)
            snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        end
        
        iter_ = caffe_solver.iter();
    end
    
    % final snapshot
    snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function check_gpu_memory(conf, caffe_solver, num_classes, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  

    % generate pseudo training data with max size
    num_channels = size(conf.image_means, 3);
    im_blob = single(zeros(max(conf.scales), conf.max_size, num_channels, conf.ims_per_batch));
    rois_blob = single(repmat([0; 0; 0; max(conf.scales)-1; conf.max_size-1], 1, conf.batch_size));
    rois_blob = permute(rois_blob, [3, 4, 1, 2]);
    labels_blob = single(ones(conf.batch_size, 1));
    labels_blob = permute(labels_blob, [3, 4, 2, 1]);
    bbox_targets_blob = zeros(4 * (num_classes+1), conf.batch_size, 'single');
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 1, 2])); 
    bbox_loss_weights_blob = bbox_targets_blob;
    
    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
    if (length(caffe_solver.net.inputs) == 6)
        % should supply bbox_loss_weights_outside
        net_inputs{end+1} = bbox_loss_weights_blob;
    end
    % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);

    if do_val
        % use the same net with train to save memory
        caffe_solver.net.set_phase('test');
        caffe_solver.net.forward(net_inputs);
        caffe_solver.net.set_phase('train');
    end
end

function model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)

    model_path = fullfile(cache_dir, file_name);
    bbox_pred_layer_name = 'bbox_pred';
    if any(ismember(caffe_solver.net.layer_names, bbox_pred_layer_name))
        bbox_stds_flatten = reshape(bbox_stds', [], 1);
        bbox_means_flatten = reshape(bbox_means', [], 1);

        % merge bbox_means, bbox_stds into the model

        weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
        biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
        weights_back = weights;
        biase_back = biase;
    
        switch ndims(weights)
            case 2
                % fc layer, weights = 4k x N (k = #classes)
                weights = ...
                    bsxfun(@times, weights, bbox_stds_flatten');     % weights = weights * stds; 
            case 4
                % conv layer, weights = M x N x C x 4k
                weights = ...
                    bsxfun(@times, weights, reshape(bbox_stds_flatten, [1 1 1 numel(bbox_stds_flatten)]));     % weights = weights * stds; 
        end
        biase = ...
            biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

        caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
        caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

        caffe_solver.net.save(model_path);
        fprintf('Saved as %s\n', model_path);

        % restore net to original state
        caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
        caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
    else
        % no bbox regression part, just save the model
        caffe_solver.net.save(model_path);
        fprintf('Saved as %s\n', model_path);
    end
end

function perf = show_state(iter, train_results, val_results)
    perf = struct;
    perf.train = struct;
    perf.train.error_cls = 1 - mean(train_results.accuarcy.data);
    perf.train.loss_cls = mean(train_results.loss_cls.data);
    perf.train.loss_reg = mean(train_results.loss_bbox.data);
    
    perf.test = struct;
    perf.test.error_cls = 1 - mean(val_results.accuarcy.data);
    perf.test.loss_cls = mean(val_results.loss_cls.data);
    perf.test.loss_reg = mean(val_results.loss_bbox.data);
    
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
        perf.train.error_cls, perf.train.loss_cls, perf.train.loss_reg);
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
            perf.test.error_cls, perf.test.loss_cls, perf.test.loss_reg);
    end
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb, ims_per_batch, batch_size)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        shuffled_inds = [];
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb, 'UniformOutput', true);
        hori_image_inds = hori_image_inds(:);
        image_indices = {find(hori_image_inds), find(~hori_image_inds)};
        % looking for images with at least one roi of any class
        % in each batch we will choose at least half such images
        pos_image_indices = arrayfun(@(x) nnz(x.class) > 0, image_roidb);
        pos_image_indices = pos_image_indices(:);
        
        % number of samples per image
        num_samples = arrayfun(@(x) sum(x.class == 0), image_roidb);

        % seperate generation for horizontal/vertical
        for k = 1:2
            image_indices_ = image_indices{k}(:);
            if isempty(image_indices_)
                continue;
            end
            
            % keep balanced batches
            image_indices_pos = find(pos_image_indices(image_indices_));
            pos_frac = length(image_indices_pos) / length(image_indices_);
            if (pos_frac < 0.5)
                % replicate "positive" images in order to have balanced sampling
                rep_factor = 1 / pos_frac - 2;
                add_pos = round(rep_factor * sum(pos_image_indices(image_indices_)));
                
                image_indices_pos_rep = repmat(image_indices_pos, ceil(rep_factor), 1);
                image_indices_ = [image_indices_; ...
                    image_indices_pos_rep(randperm(length(image_indices_pos_rep), add_pos))];
            else
                % replicate "negative" images in order to have balanced sampling
            end
            
            % split to groups by num_samples
            [C, ~, ic] = unique(num_samples(image_indices_));
            for i_group = 1:length(C)
                image_indices__ = image_indices_(ic == i_group);
                ims_per_batch_ = ims_per_batch;
                if (C(i_group)*ims_per_batch_ < batch_size)
                    % we don't have enough samples to reach batch_size
                    ims_per_batch_ = round(batch_size / C(i_group));
                end
                
                % random perm
                lim = floor(length(image_indices__) / ims_per_batch_) * ims_per_batch_;
                image_indices__ = image_indices__(randperm(length(image_indices__), lim));
                % combine sample for each ims_per_batch 
                image_indices__ = reshape(image_indices__, ims_per_batch_, []);
                shuffled_inds = [shuffled_inds, num2cell(image_indices__, 1)];
            end
            
            % shuffle batches order
            shuffled_inds = shuffled_inds(randperm(length(shuffled_inds)));
        end
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        % assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end