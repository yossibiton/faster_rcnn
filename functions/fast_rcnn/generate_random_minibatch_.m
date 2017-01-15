function [shuffled_inds, sub_inds] = generate_random_minibatch_(shuffled_inds, image_roidb, ims_per_batch, batch_size, opts)
    
    if ~exist('batch_size', 'var')
        batch_size = ims_per_batch;
    end
    if ~exist('opts', 'var')
        opts = struct('use_weights', false, ...
            'pos_part', 0.5);
    end
    
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

        for k = 1:2
            % 01 - seperate generation for horizontal/vertical
            image_indices1 = image_indices{k}(:);
            if isempty(image_indices1)
                continue;
            end
            
            epoch_size = floor(length(image_indices1) / ims_per_batch)*ims_per_batch;
            
            % 02 - balance classes
            image_indices2 = balance_classes(image_indices1, pos_image_indices, opts.pos_part);
            
            % 03 (disabled) - split to groups by num_samples
            if false
                % number of samples per image
                num_samples = arrayfun(@(x) sum(x.class == 0), image_roidb);
        
                image_indices3 = [];
                [C, ~, ic] = unique(num_samples(image_indices2));
                for i_group = 1:length(C)
                    image_indices3_ = image_indices2(ic == i_group);
                    ims_per_batch_ = ims_per_batch;
                    if (C(i_group)*ims_per_batch_ < batch_size)
                        % we don't have enough samples to reach batch_size
                        ims_per_batch_ = round(batch_size / C(i_group));
                    end

                    % random perm
                    lim = floor(length(image_indices3_) / ims_per_batch_) * ims_per_batch_;
                    image_indices3_ = image_indices3_(randperm(length(image_indices3_), lim));
                    % combine sample for each ims_per_batch 
                    image_indices3 = [image_indices3, reshape(image_indices3_, ims_per_batch_, [])];
                end
            else
                image_indices3 = image_indices2;
            end
            
            % 04 - change samples frequency according to weights
            if opts.use_weights
                weights = [image_roidb.weight];
                image_indices4 = balance_weights(image_indices3, weights);
            else
                image_indices4 = image_indices3;
            end
            
            % Final - shiffle examples & pack into batches
            image_indices_final = image_indices4(randperm(length(image_indices4), epoch_size));
            image_indices_final = reshape(image_indices_final, ims_per_batch, []);
            shuffled_inds = num2cell(image_indices_final, 1);
        end
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        % assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function image_indices = balance_classes(image_indices, pos_image_indices, pos_part)
    % pos_part = [0..1] 
    %         the relative pos part in each minibatch
    
    if isnan(pos_part)
        % keep original ratio
        return;
    end
    
    image_indices_pos = find(pos_image_indices(image_indices));
    pos_frac = length(image_indices_pos) / length(image_indices);
    if (pos_frac < pos_part)
        % replicate "positive" images in order to have balanced sampling
        rep_factor = 1 / pos_frac - 1/pos_part;
        add_pos = round(rep_factor * sum(pos_image_indices(image_indices)));

        image_indices_pos_rep = repmat(image_indices_pos, ceil(rep_factor), 1);
        image_indices = [image_indices; ...
            image_indices_pos_rep(randperm(length(image_indices_pos_rep), add_pos))];
    else
        % replicate "negative" images in order to have balanced sampling
    end
end

function image_indices_out = balance_weights(image_indices, weights)
    % qunatize weights into integers
    min_weight = min(weights);
    weights_q = round(weights / min_weight);
    [weights_u, ~, weights_u_inds] = unique(weights_q);
    
    % duplicate each example according to its weight
    image_indices_out = [];
    for i_weight = 1:length(weights_u)
        weight_value = weights_u(i_weight);
        weight_indices = (weights_u_inds == i_weight);
        
        image_indices_out = [image_indices_out; ...
            repmat(image_indices(weight_indices), [weight_value 1])];
    end
end