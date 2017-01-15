function image_roidb = bboxreg_normalize_targets(image_roidb, means, stds, unnormalize)

if ~exist('unnormalize', 'var')
    unnormalize = false;
end
    
num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
num_classes = size(image_roidb(1).overlap, 2);

for i = 1:num_images
    targets = image_roidb(i).bbox_targets;
    for cls = 1:num_classes
        cls_inds = find(targets(:, 1) == cls);
        if ~isempty(cls_inds)
            if unnormalize
                % unnormalize : multiply by std and then add mean
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@times, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@plus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
            else
                % normalize : subtract mean and then divide by std
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
            end
        end
    end
end