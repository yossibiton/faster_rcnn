function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb, ...
    ims_per_batch, batch_size)

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

        % 1 - seperate generation for horizontal/vertical
        for k = 1:2
            image_indices1 = image_indices{k}(:);
            if isempty(image_indices1)
                continue;
            end
            
            % 1.1 - replicating pos samples (to create balanced batches)
            image_indices2 = balance_classes(image_indices1, pos_image_indices);
            
            % split to groups by num_samples
            if false
                [C, ~, ic] = unique(num_samples(image_indices2));
                for i_group = 1:length(C)
                    image_indices3 = image_indices2(ic == i_group);
                    ims_per_batch_ = ims_per_batch;
                    if (C(i_group)*ims_per_batch_ < batch_size)
                        % we don't have enough samples to reach batch_size
                        ims_per_batch_ = round(batch_size / C(i_group));
                    end

                    % random perm
                    lim = floor(length(image_indices3) / ims_per_batch_) * ims_per_batch_;
                    image_indices3 = image_indices3(randperm(length(image_indices3), lim));
                    % combine sample for each ims_per_batch 
                    image_indices3 = reshape(image_indices3, ims_per_batch_, []);
                    shuffled_inds = [shuffled_inds, num2cell(image_indices3, 1)];
                end
            else
                lim = floor(length(image_indices2) / ims_per_batch) * ims_per_batch;
                image_indices3 = image_indices2(randperm(length(image_indices2), lim));
                % combine sample for each ims_per_batch 
                image_indices3 = reshape(image_indices3, ims_per_batch, []);
                shuffled_inds = num2cell(image_indices3, 1);
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

function image_indices = balance_classes(image_indices, pos_image_indices)
    image_indices_pos = find(pos_image_indices(image_indices));
    pos_frac = length(image_indices_pos) / length(image_indices);
    if (pos_frac < 0.5)
        % replicate "positive" images in order to have balanced sampling
        rep_factor = 1 / pos_frac - 2;
        add_pos = round(rep_factor * sum(pos_image_indices(image_indices)));

        image_indices_pos_rep = repmat(image_indices_pos, ceil(rep_factor), 1);
        image_indices = [image_indices; ...
            image_indices_pos_rep(randperm(length(image_indices_pos_rep), add_pos))];
    else
        % replicate "negative" images in order to have balanced sampling
    end
end