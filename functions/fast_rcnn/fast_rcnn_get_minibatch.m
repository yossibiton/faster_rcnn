function [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] = fast_rcnn_get_minibatch(conf, image_roidb, prefetch)
% [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] ...
%    = fast_rcnn_get_minibatch(conf, image_roidb)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    % Sample random scales to use for each image in this batch
    if conf.keep_scale
        img_sizes = reshape([image_roidb.im_size]', [2 length(image_roidb)]);
        batch_scales = min(img_sizes, [], 1);
    else
        random_scale_inds = randi(length(conf.scales), num_images, 1);
        batch_scales = conf.scales(random_scale_inds);
    end
    
    assert(mod(conf.batch_size, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
    
    % Get the input image blob
    if prefetch
        [im_blob, im_scales] = get_image_blob_prefetch(conf, image_roidb, batch_scales);
    else
        [im_blob, im_scales] = get_image_blob(conf, image_roidb, batch_scales);
    end
    
    % build the region of interest and label blobs
    rois_blob = zeros(0, 5, 'single');
    labels_blob = zeros(0, 1, 'single');
    bbox_targets_blob = zeros(0, 4 * (num_classes+1), 'single');
    bbox_loss_blob = zeros(size(bbox_targets_blob), 'single');
    
    for i = 1:num_images
        [labels, ~, im_rois, bbox_targets, bbox_loss] = ...
            sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image);
        
        % Add to ROIs blob
        feat_rois = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scales(i));
        batch_ind = i * ones(size(feat_rois, 1), 1);
        rois_blob_this_image = [batch_ind, feat_rois];
        rois_blob = [rois_blob; rois_blob_this_image];
        
        % Add to labels, bbox targets, and bbox loss blobs
        labels_blob = [labels_blob; labels];
        bbox_targets_blob = [bbox_targets_blob; bbox_targets];
        bbox_loss_blob = [bbox_loss_blob; bbox_loss];
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    if size(im_blob, 3) == 3
        im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    end
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
    labels_blob = single(permute(labels_blob, [3, 4, 2, 1]));
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 2, 1])); 
    bbox_loss_blob = single(permute(bbox_loss_blob, [3, 4, 2, 1]));
    
    assert(~isempty(im_blob));
    assert(~isempty(rois_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
end

%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales] = get_image_blob(conf, images, target_sizes)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        if isempty(images(i).image)
            im = imread(images(i).image_path);
        else
            im = images(i).image;
        end
        
        if ndims(im) == 2 && ndims(conf.image_means) == 3
            % replicate from grayscale to "rgb"
            im = cat(3, im, im, im);
        end
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_sizes(i), conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

function [im_blob, im_scales] = get_image_blob_prefetch(conf, images, target_sizes)
    
    num_images = length(images);
    
    % In this function we don't do scaling at all
    % TODO : add subtract mean (currently assume grayscale)
    prefetch_args = {'SubtractAverage', conf.image_means(1)};
    processed_ims = vl_imreadjpeg(...
        image_roidb_train(sub_inds_next).image_path, prefetch_args{:});
    im_scales = ones(num_images, 1);
    
    if false
    for i = 1:num_images
        if isempty(images(i).image)
            im = imread(images(i).image_path);
        else
            im = images(i).image;
        end
        
        if ndims(im) == 2 && ndims(conf.image_means) == 3
            % replicate from grayscale to "rgb"
            im = cat(3, im, im, im);
        end
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_sizes(i), conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    end
    
    im_blob = im_list_to_blob(processed_ims);
end
%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, overlaps, rois, bbox_targets, bbox_loss_weights] = ...
    sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image)

    [overlaps, labels] = max(image_roidb.overlap, [], 2);
%     labels = image_roidb(1).max_classes;
%     overlaps = image_roidb(1).max_overlaps;
    rois = image_roidb.boxes;
    
    % Select foreground ROIs as those with >= FG_THRESH overlap, 
    %   and which are not gt bbox
    fg_inds = find(overlaps >= conf.fg_thresh & image_roidb.class == 0);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image = min(fg_rois_per_image, length(fg_inds));
    % Sample foreground regions without replacement
    if ~isempty(fg_inds)
       fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
    bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
    % Sample foreground regions without replacement
    if ~isempty(bg_inds)
       bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
    end
    % The indices that we're selecting (both fg and bg)
    keep_inds = [fg_inds; bg_inds];
    % Select sampled values from various arrays
    labels = labels(keep_inds);
    % Clamp labels for the background ROIs to 0
    labels((fg_rois_per_this_image+1):end) = 0;
    overlaps = overlaps(keep_inds);
    rois = rois(keep_inds, :);
    
    assert(all(labels == image_roidb.bbox_targets(keep_inds, 1)));
    
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb.overlap, 2);
    
    [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, ...
        image_roidb.bbox_targets(keep_inds, :), num_classes);
end

function [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, bbox_target_data, num_classes)
%% Bounding-box regression targets are stored in a compact form in the roidb.
 % This function expands those targets into the 4-of-4*(num_classes+1) representation used
 % by the network (i.e. only one class has non-zero targets).
 % The loss weights are similarly expanded.
% Return (N, (num_classes+1) * 4, 1, 1) blob of regression targets
% Return (N, (num_classes+1 * 4, 1, 1) blob of loss weights
    clss = bbox_target_data(:, 1);
    bbox_targets = zeros(length(clss), 4 * (num_classes+1), 'single');
    bbox_loss_weights = zeros(size(bbox_targets), 'single');
    inds = find(clss > 0);
    for i = 1:length(inds)
       ind = inds(i);
       cls = clss(ind);
       targets_inds = (1+cls*4):((cls+1)*4);
       bbox_targets(ind, targets_inds) = bbox_target_data(ind, 2:end);
       bbox_loss_weights(ind, targets_inds) = 1;  
    end
end


