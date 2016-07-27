function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb, ...
    ims_per_batch, batch_size)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        shuffled_inds = [];
        
        % make sure each minibatch only has horizontal or vertical images, 
        % to save gpu memory
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb, 'UniformOutput', true);
        hori_image_inds = hori_image_inds(:);
        image_indices = {find(hori_image_inds), find(~hori_image_inds)};
        
        % tagging images with at least one roi of as "positives", so
        % in each minibatch we will choose at least half such images
        class_image_indices = arrayfun(@(x) nnz(x.class) > 0, image_roidb);
        class_image_indices = class_image_indices(:);
        
        % number of samples per image
        num_samples = arrayfun(@(x) sum(x.class == 0), image_roidb);

        % 1 - seperate generation for horizontal/vertical
        for k = 1:2
            image_indices2 = image_indices{k}(:);
            if isempty(image_indices2)
                continue;
            end
            
            % 1.1 - replicating pos/neg samples (to create balanced batches)
            image_indices2 = balance_classes(...
                image_indices2, class_image_indices(image_indices2));
            n_images_class = length(image_indices2{1});
            
            % 1.2 - changing samples distribution according to their weights
            %       (replication for pos/neg seperately without changing their ratio)
            image_indices3 = cell(1, 2);
            for i_class = 1:length(image_indices2)
                image_indices_class = image_indices2{i_class};
                image_indices_class_new = [];
                weights_class = [image_roidb(image_indices_class).weight];
                % assuming weight are positive integers out of limited set
                % of values (1:N)
                [W, ~, iw] = unique(weights_class);
                for i_weight = 1:length(W)
                    weight_value = W(i_weight);
                    indices_rep = repmat(image_indices_class(iw == i_weight), [1 weight_value]);
                    image_indices_class_new = [image_indices_class_new; indices_rep(:)];
                end
                % we want to save original numer of samples in batch
                image_indices_class_new = image_indices_class_new(...
                    randperm(length(image_indices_class_new), n_images_class));
                image_indices3{i_class} = image_indices_class_new;
            end
            
            % pack into minibatches, where each has half positives half
            % negatives samples
            shuffled_inds_ = [];
            for i_class = 1:length(image_indices3)
                n_images = (ims_per_batch/2) * ...
                    floor(length(image_indices3{i_class}) / (ims_per_batch/2));
                shuffled_inds_ = [shuffled_inds_; ...
                    reshape(image_indices3{i_class}(1:n_images), ims_per_batch/2, [])];
            end
            shuffled_inds = [shuffled_inds, num2cell(shuffled_inds_, 1)];
        
            % 1.3 - split to groups by num_samples (disabled right now)
            %image_indices3 = split_num_samples(...
            %    image_indices2, num_samples(image_indices2), ims_per_batch, batch_size);
            %shuffled_inds = [shuffled_inds, image_indices3];
        end
        
        % shuffle minibatches order
        shuffled_inds = shuffled_inds(randperm(length(shuffled_inds)));
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        % assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function indices = balance_classes(indices, classes)
    for class = 0:1
        % loop over 2 cases : 0 = negatives, 1 = positives
        %   but we should replicate only one class samples (0/1)
        curr_class_image_indices = find(classes == class);
        class_frac = length(curr_class_image_indices) / length(indices);
        if (class_frac < 0.5)
            % replicate current class images in order to have balanced sampling
            rep_factor = 1 / class_frac - 2;
            add_class = round(rep_factor * length(curr_class_image_indices));
            image_indices_class_rep = repmat(curr_class_image_indices, ceil(rep_factor), 1);
            image_indices_class_add = image_indices_class_rep(randperm(length(image_indices_class_rep), add_class));

            indices = [indices; image_indices_class_add];
            break;
        end
    end
    neg_indices = classes == 0;
    indices = {indices(neg_indices), indices(~neg_indices)};
    % eventualy we want to get the exact same number of positive/negatives examples
    n_images_class = min(length(indices{1}), length(indices{2}));
    indices = {indices{1}(1:n_images_class), indices{2}(1:n_images_class)};
end

function indices_splitted = split_num_samples(indices, num_samples, ...
    ims_per_batch, batch_size)
    [C, ~, ic] = unique(num_samples);
    indices_splitted = cell(length(C), 1);
    for i_group = 1:length(C)
        indices_group = indices(ic == i_group);
        ims_per_batch_ = ims_per_batch;
        if (C(i_group)*ims_per_batch_ < batch_size)
            % we don't have enough samples to reach batch_size
            ims_per_batch_ = round(batch_size / C(i_group));
        end

        % random perm and pack into batches
        lim = floor(length(indices_group) / ims_per_batch_) * ims_per_batch_;
        indices_group = indices_group(randperm(length(indices_group), lim));
        indices_group = reshape(indices_group, ims_per_batch_, []);
        indices_splitted{i_group} = num2cell(indices_group, 1);
    end
end