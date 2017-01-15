function [adaptive_opts, image_roidb_train, use_weights, image_roidb_train_loaded] = adaptive_training_op(iter, adaptive_opts, image_roidb_train, image_roidb_train_loaded)

adaptive_mode = adaptive_opts.adaptive_mode;

use_weights = false;
if isempty(adaptive_opts.start_point)
    % (A) already exceeded the adaptive phase start point
    use_weights = (adaptive_mode == 5);
    return;
end
if (iter < adaptive_opts.start_point)
    % (B) not exceededed yet the adaptive phase start point
    return;
end
% (C) first time exceededed the adaptive phase start point
adaptive_opts.start_point = adaptive_opts.start_point(2:end);
use_weights = (adaptive_mode == 5);

is_pos = arrayfun(@(x) nnz(x.class) > 0, image_roidb_train);
scores = [image_roidb_train.score]';

% scroes distribution for pos/neg
scores_bar = min(scores):0.1:max(scores);
pos_pdf = hist(scores(is_pos) , scores_bar);
neg_pdf = hist(scores(~is_pos), scores_bar);
pos_cdf = cumsum(pos_pdf) / sum(pos_pdf);
neg_cdf = cumsum(neg_pdf) / sum(neg_pdf);

% (1) find where the positives cdf exceeds 0.1% / 0%
% (2) find where the negatives cdf exceeds 50% of num_postives
switch adaptive_mode
    case 1
        k1 = find(pos_cdf >  0.001, 1);
        k2 = find(neg_cdf >= (1 - sum(is_pos) / sum(neg_pdf)), 1);
        score_th = scores_bar(min(k1, k2));
    case 2
        k1 = find(pos_cdf >  0, 1);
        k2 = find(neg_cdf >= (1 - sum(is_pos) / sum(neg_pdf)), 1);
        score_th = scores_bar(min(k1, k2));
    case {3, 4, 5} % eliminate 10% og negatives, also lower lr in these case
        k1 = find(pos_cdf >  0, 1);
        k2 = find(neg_cdf >= 0.1, 1);
        score_th = scores_bar(min(k1, k2));
end
figure; hold; 
plot(scores_bar, neg_pdf, 'r'); plot(scores_bar, pos_pdf, 'g');
plot(scores_bar(k1)*ones(1, 2), [0 max([neg_pdf(:); pos_pdf(:)])], 'k');
plot(scores_bar(k2)*ones(1, 2), [0 max([neg_pdf(:); pos_pdf(:)])], 'b');

% filter out negative examples with really low score
indices_eliminate = (~is_pos) & (scores < score_th);

if (adaptive_mode == 5)
    % not really eliminating examples, just giving them 0.5 weight
    x1 = repmat({1}, 1, sum(indices_eliminate));
    x2 = repmat({2}, 1, length(indices_eliminate) - length(x1));

    [image_roidb_train(indices_eliminate).weight ] = x1{:};
    [image_roidb_train(~indices_eliminate).weight] = x2{:};
else
    % removing out chosen examples
    image_roidb_train(indices_eliminate) = [];
    image_roidb_train_loaded.im_blob(indices_eliminate) = [];
    image_roidb_train_loaded.im_scales(indices_eliminate) = [];

    % recalculate bbox_means & bbox_stds (not really needed because
    % we didnt change positive examples)
    %image_roidb_train = bboxreg_normalize_targets(image_roidb_train, ...
    %    bbox_means, bbox_stds, true);
    %[bbox_means, bbox_stds] = bboxreg_get_targets_stats(image_roidb_train);
    %image_roidb_train = bboxreg_normalize_targets(...
    %    image_roidb_train, bbox_means, bbox_stds);
end