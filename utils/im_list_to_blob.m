function blob = im_list_to_blob(ims)
    max_shape = max(cell2mat(cellfun(@size, ims(:), 'UniformOutput', false)), [], 1);
    
    num_channels = cellfun(@(x) size(x, 3), ims, 'UniformOutput', true);
    assert(all(num_channels == 3) || all(num_channels == 1));
    num_channels = num_channels(1);
    
    num_images = length(ims);
    
    blob = zeros(max_shape(1), max_shape(2), num_channels, num_images, 'single');
    
    for i = 1:length(ims)
        im = ims{i};
        blob(1:size(im, 1), 1:size(im, 2), :, i) = im; 
    end
end