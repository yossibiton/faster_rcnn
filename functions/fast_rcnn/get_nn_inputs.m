function net_inputs = get_nn_inputs(conf, image_roidb, image_roidb_loaded, sub_db_inds, opts)
    if opts.datatset_in_memory
        im_blob = image_roidb_loaded.im_blob(sub_db_inds);
        % reshape to W x H x D x batch_size
        im_blob = cell2mat(reshape(im_blob, [1 1 1 length(im_blob)]));

        im_scales = image_roidb_loaded.im_scales(sub_db_inds);
        [im_blob, ~, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
            fast_rcnn_get_minibatch(conf, image_roidb(sub_db_inds), opts.prefetch, ...
            im_blob, im_scales);
    else
        [im_blob, ~, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
            fast_rcnn_get_minibatch(conf, image_roidb(sub_db_inds), opts.prefetch);
    end
    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, ...
        bbox_loss_weights_blob, ones(size(bbox_loss_weights_blob))};
end