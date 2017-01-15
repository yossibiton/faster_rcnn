function [means, stds] = bboxreg_get_targets_stats(image_roidb)

num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
num_classes = size(image_roidb(1).overlap, 2);
    
% Compute values needed for means and stds
% var(x) = E(x^2) - E(x)^2
class_counts = zeros(num_classes, 1) + eps;
sums = zeros(num_classes, 4);
squared_sums = zeros(num_classes, 4);
for i = 1:num_images
   targets = image_roidb(i).bbox_targets;
   for cls = 1:num_classes
      cls_inds = find(targets(:, 1) == cls);
      if ~isempty(cls_inds)
         class_counts(cls) = class_counts(cls) + length(cls_inds); 
         sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
         squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
      end
   end
end

means = bsxfun(@rdivide, sums, class_counts);
stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;

% add background class
means = [0, 0, 0, 0; means]; 
stds = [0, 0, 0, 0; stds];