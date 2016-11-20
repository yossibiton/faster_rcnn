cp setup/Makefile.config external/caffe
cd external/caffe
git apply ../../setup/0001-upgrading-matlab-interface-by-adding-new-methods-to-.patch

echo "add manually -D_FORCE_INLINES to NVCCFLAGS in external/caffe/Makefile"
echo " add the following lines to caffe/src/caffe/util/upgrade_proto.cpp, function UpgradeV1LayerType :
  case V1LayerParameter_LayerType_RESHAPE:
    return "Reshape";
  case V1LayerParameter_LayerType_ROIPOOLING:
    return "ROIPooling";
  case V1LayerParameter_LayerType_SMOOTH_L1_LOSS:
    return "SmoothL1Loss";
    "