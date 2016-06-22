function active_caffe_mex(gpu_id)
% active_caffe_mex(gpu_id, caffe_version)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    % set gpu in matlab
    gpuDevice(gpu_id);
    cur_dir = pwd;
    caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');

    
    addpath(genpath(caffe_dir));
    cd(caffe_dir);
    caffe.set_device(gpu_id-1);
    cd(cur_dir);
end
