target_dir=../caffe_Austriker
git clone -b fast-rcnn https://github.com/Austriker/caffe $target_dir
cp setup/0001-upgrading-matlab-interface-by-adding-new-methods-to-.patch $target_dir/matlab_interface.patch
cp setup/Makefile.config $target_dir
cd $target_dir
git apply matlab_interface.patch

echo "Please build caffe version in $target_dir (you can use the script install_prerequisites.sh)"
echo "Recommended prerequisites : OpenBLAS, cuda-8.0, cudnn-5.1"
    
