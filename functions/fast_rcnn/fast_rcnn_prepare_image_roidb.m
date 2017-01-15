function [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
%   Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------- 
    
    if ~exist('bbox_means', 'var')
        bbox_means = [];
        bbox_stds = [];
    end
    
    if ~iscell(imdbs)
        imdbs = {imdbs};
        roidbs = {roidbs};
    end

    imdbs = imdbs(:);
    roidbs = roidbs(:);
    
    % we should create imroi_db (combination of imdb & roidb)
    image_roidb = {};
    for k = 1:length(imdbs)
        x = imdbs{k};
        y_rois = roidbs{k}.rois;
        roidb_fields = fields(roidbs{1}.rois);
        if all(ismember({'image_path', 'image_id', 'im_size', 'imdb_name'}, roidb_fields))
            % roidb has laready has all fields we need for creating imroi_db
            y_rois().image = [];
            y_rois().bbox_targets = [];
            image_roidb{k} = y_rois;
        else
            n_images = length(x.image_ids);
            image_roidb_i = struct('image_path', cell(1, n_images), 'image_id', cell(1, n_images), 'im_size', cell(1, n_images), ...
                'imdb_name', cell(1, n_images), 'overlap', cell(1, n_images), 'boxes', cell(1, n_images), 'class', cell(1, n_images), ...
                'image', cell(1, n_images), 'bbox_targets', cell(1, n_images));
            parfor z = 1:n_images
                image_roidb_i(z) = struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                    'overlap', y_rois(z).overlap, 'boxes', y_rois(z).boxes, 'class', y_rois(z).class, 'image', [], 'bbox_targets', []);
            end
            image_roidb{k} = image_roidb_i';
        end
    end
    image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end

function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb(i).boxes; 
       [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb(i).overlap);
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end
        
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        [means, stds] = bboxreg_get_targets_stats(image_roidb);
    end
    
    % Normalize targets
    image_roidb = bboxreg_normalize_targets(image_roidb, means, stds);
end


function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)

    overlap = full(overlap);

    [max_overlaps, max_labels] = max(overlap, [], 2);

    % ensure ROIs are floats
    rois = single(rois);
    
    bbox_targets = zeros(size(rois, 1), 5, 'single');
    
    % Indices of ground-truth ROIs
    gt_inds = find(max_overlaps == 1);
    
    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps >= conf.bbox_thresh);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));

        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    end
    
    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps >= conf.fg_thresh;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;
    
    % check if there is any fg or bg sample. If no, filter out this image
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end