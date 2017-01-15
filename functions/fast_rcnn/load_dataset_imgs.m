function image_roidb_loaded = load_dataset_imgs(conf, image_roidb, cache_path)

    if exist(cache_path, 'file')
        image_roidb_loaded = load(cache_path);
        if conf.image_means(1) ~= image_roidb_loaded.image_mean
            for i_image = 1:length(image_roidb_loaded.im_blob)
                image_roidb_loaded.im_blob{i_image} = ...
                    image_roidb_loaded.im_blob{i_image} + ...
                image_roidb_loaded.image_mean - ...
                conf.image_means(1);
            end
        end
    else
        batch_size = conf.batch_size;
        n_batches = ceil(length(image_roidb) / batch_size);
        im_blob = cell(n_batches, 1);
        im_scales = cell(n_batches, 1);
        parfor i_batch = 1:n_batches
            batch_start = 1 + (i_batch - 1)*batch_size;
            batch_end = min(length(image_roidb), ...
                batch_start + batch_size - 1);
            batch_indices = batch_start:batch_end;
            batch_size_actual = length(batch_indices);
            if (batch_size_actual < batch_size)
                batch_indices_full = ...
                    [batch_indices, 1:(batch_size - batch_size_actual)];
            else
                batch_indices_full = batch_indices; 
            end

            [im_blob_batch, im_scales_batch] = ...
                fast_rcnn_get_minibatch(conf, image_roidb(batch_indices_full));
            im_blob_batch = squeeze(mat2cell(im_blob_batch, ...
                size(im_blob_batch, 1), size(im_blob_batch, 2), ...
                size(im_blob_batch, 3), ones(1, conf.batch_size)));

            im_blob{i_batch} = im_blob_batch(1:batch_size_actual);
            im_scales{i_batch} = im_scales_batch(1:batch_size_actual);
        end
        im_blob = cat(1, im_blob{:});
        im_scales = cat(1, im_scales{:});
        image_mean = conf.image_means(1);
        save(cache_path, 'im_blob', 'im_scales', 'image_mean', '-v7.3');
        
        % trasnforming into struct
        image_roidb_loaded = struct;
        image_roidb_loaded.im_blob = im_blob;
        image_roidb_loaded.im_scales = im_scales;
        
        clear im_blob im_scales
    end
end
