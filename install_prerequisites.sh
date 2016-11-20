# OpenBLAS
git clone https://github.com/xianyi/OpenBLAS ~/OpenBLAS
pushd ~/OpenBLAS
make
sudo make install

# CUDA 8.0
sudo /server/software/NVIDIA/cuda/cuda_8.0.44_linux.run

# CUDNN 5.1
cd /usr/local
sudo tar xvf /server/software/NVIDIA/cudnn/cudnn-8.0-linux-x64-v5.1.tgz

# Caffe
popd
cd ../caffe_Austriker
make all
make matcaffe